import numpy as np
from dataclasses import replace
import pandas as pd
from loguru import logger
from rich.progress import track
from natsort import natsorted
from pathlib import Path
import pywt

from .creating_dataset import init_dataset
# from .augmentations import get_default_transform
from . import creating_dataset
from .creating_dataset import LEFT_TO_RIGHT_HAND

from abcidatasets import Dataset, DatasetVariable, DatasetSession
from abcidatasets.session import DatasetTrial
from abcidatasets.dataset.utils import make_acausal_kernel

DATA_PATH = "./dataset_v2_blocks"
LEFT_TO_RIGHT_HAND = [6, 5, 4, 3, 2, 1, 0, 7]
n_inputs, n_outputs = 8, 20


def smooth(x, kernel_size=5):
    kernel = make_acausal_kernel(kernel_size)
    kernel = kernel / np.sum(kernel)
    for i in range(x.shape[0]):
        x[i] = np.convolve(np.abs(x[i]), kernel, mode='same')
    return x
    

def load_submission_dataset():
    test_data_name = 'fedya_tropin_standart_elbow_left'  # shoould match `test_dataset_list` used to train the model

    dp = Path(DATA_PATH) / 'dataset_v2_blocks'
    data_folder = dp / "amputant" / "left" / test_data_name / "preproc_angles" / "submit"
    all_paths = natsorted(data_folder.glob('*.npz'))
    
    trials_data = []
    for p in all_paths:
        sample = np.load(p, allow_pickle=True)
        myo = sample['data_myo'][:, LEFT_TO_RIGHT_HAND]
        trials_data.append(myo)

    submit_gt = pd.read_csv(dp/'submit_gt.csv')
    return trials_data, submit_gt



def emg_amplitude(x):
    return smooth(np.abs(x))




def numpy_fles_to_trials(path, downsample_target_factor, dt, t):
    files = natsorted(path.glob('*.npz'))
    is_left_hand = 'left' in str(path)

    trials = []
    for f in files:
        data = dict(np.load(f, allow_pickle=True))
        myo = data['data_myo']
        myo = emg_amplitude(myo.T).T[::downsample_target_factor, :]

        if is_left_hand:
            myo = myo[:, LEFT_TO_RIGHT_HAND]    

        time = np.linspace(0, myo.shape[0] / dt, myo.shape[0]) * 1000 + t 
        t = time[-1]

        trials.append(
            DatasetTrial(
                myo,
                data['data_angles'][::downsample_target_factor, :],
                time,
            )
        )

    return trials, t
    

def collect_paths(load_test_only):
    master = Path('/om2/user/claudif/DecodingAlgorithms/BCI_ALVI_challenge_competition/dataset_v2_blocks/dataset_v2_blocks')
    assert master.exists(), f"Data path {master} does not exist"
    subs = [
        master / 'amputant' / 'left',
        master / 'health' / 'left',
        master / 'health' / 'right',
    ]

    train_paths = []
    test_paths = []
    for s in subs:
        assert s.exists(), f"Data path {s} does not exist"
        subfolders = [(x/'preproc_angles') for x in s.iterdir() if x.is_dir()]
        for subfld in subfolders:
            assert  subfld.exists(), f"Data path {subfld} does not exist"

            if load_test_only and 'fedya_tropin_standart_elbow_left' not in str(subfld):
                continue

            train = subfld / 'train'
            test = subfld / 'test'

            if train.exists():
                train_paths.append(train)
            if test.exists():
                if 'fedya_tropin_standart_elbow_left' in str(test):
                    test_paths.append(test)
                # else:
                #     train_paths.append(test)

    print(f"Found {len(train_paths)} training and {len(test_paths)} test file paths in the dataset.")
    return train_paths, test_paths
                
    
def time_series_decomposition(time_series):
    # Decompose the time series using Discrete Wavelet Transform (DWT)
    wavelet = 'db4'
    coeffs = pywt.wavedec(time_series, wavelet, level=6)

    # Extract approximation (low-frequency) and detail (high-frequency) coefficients
    low = list(coeffs)  # Coefficients at the highest level (lowest frequency)
    low[1:] = [np.zeros_like(c) for c in low[1:]]  # Zero out the detail coefficients

    medium = list(coeffs)
    medium[:2] = [np.zeros_like(c) for c in medium[:2]]  # Zero out the low
    medium[4:] = [np.zeros_like(c) for c in medium[4:]]  # Zero out the low

    high = list(coeffs)
    high[:5] = [np.zeros_like(c) for c in high[:5]]  
    
    T = len(time_series)
    low_f = pywt.waverec(low, wavelet)[:T]
    med_f = pywt.waverec(medium, wavelet)[:T]
    high_F = pywt.waverec(high, wavelet)[:T]
    return low_f, med_f, high_F


def recombine_predictions(pred):
    # Recombine the predictions
    pred = pred.reshape(-1, 20, 2)
    pred = np.sum(pred, axis=2)
    return pred


def augment_features(session):
    Y = np.concatenate([t.Y for t in session.all_trials], axis=0)  # of shape T x D
    cuts = [t.Y.shape[0] for t in session.all_trials]

    T, dim = Y.shape
    Y_new = np.zeros((T, dim * 2))

    count = 0
    for i in range(dim):
        low, med, _ = time_series_decomposition(Y[:, i])
        Y_new[:, count] = low
        Y_new[:, count + 1] = med
        # Y_new[:, count + 2] = high
        count += 2

    for i, t in enumerate(session.all_trials):
        t.Y = Y_new[:cuts[i], :]
        Y_new = Y_new[cuts[i]:, :]
        assert t.Y.shape[0] == t.X.shape[0], f"Trial {i} has different shapes for {t.X.shape:=} and {t.Y.shape:=}"
        assert t.Y.shape[1] == dim  * 2, f"Expected {dim * 2} features, got {t.Y.shape[1]} for trial {i}"

    return session


def load_data_into_omen_dataset(
        n_sessions:int=-1,
        downsample_movements_factor:int=1,
        load_test_only=False,
        downsample_target_factor=1,
        ):

    train_paths, test_paths = collect_paths(load_test_only)
    print(f"Found {len(train_paths)} training and {len(test_paths)} test sessions in the dataset.")

    dt = int(1000/200) * downsample_target_factor
    variables = []
    for i in range(n_outputs):
        for tag in ('low', 'med'):
            variables.append(DatasetVariable(f'angle_{i}_{tag}', 'hand', False))

    omen_ds =  Dataset('hand', '', -1, dt, 1, variables, [])

    t = 0
    for i, path in enumerate(train_paths):
        if n_sessions > 0 and i >= n_sessions:
            break
            
        name = path.parent.parent.stem
        print(name)
        session = DatasetSession(name, omen_ds.name, dt, variables, [], [], [])
        trials, t = numpy_fles_to_trials(path, downsample_target_factor, dt, t)
        session.train_trials.extend(trials)

        # look for a test path with the same name
        found = False
        for test_path in test_paths:
            if name in test_path.parent.parent.stem:
                found = True
                trials, t = numpy_fles_to_trials(test_path, downsample_target_factor, dt, t)
                session.test_trials.extend(trials)
                break
        if not found:
            session.test_trials = session.train_trials[-3:]

        session = augment_features(session)
        omen_ds.sessions.append(session)
    # print(f"Amount of data: {len(session.train_trials)} train and {len(session.test_trials)} test trials")

    # omen_ds.sessions.append(session)
    return omen_ds
