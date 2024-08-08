import numpy as np
from dataclasses import replace
import pandas as pd
from loguru import logger
from rich.progress import track
from natsort import natsorted
from pathlib import Path

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

def reshape_and_average(x, n):
    """
        x is emg data of shape n_inputs x T. 
        The function should return an array of shape n_inputs x T//N
        where each value is the average of the corresponding N values in x
    """
    return x[:, ::n]
    



def numpy_fles_to_trials(path, downsample_target_factor, dt, t):
    files = natsorted(path.glob('*.npz'))
    is_left_hand = 'left' in str(path)

    trials = []
    for f in files:
        data = dict(np.load(f, allow_pickle=True))
        myo = data['data_myo'][::downsample_target_factor, :]

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
            train = subfld / 'train'
            test = subfld / 'test'

            if train.exists():
                if load_test_only and 'fedya_tropin_standart_elbow_left' in str(train):
                    continue
                train_paths.append(train)
            if test.exists():
                if 'fedya_tropin_standart_elbow_left' in str(test):
                    test_paths.append(test)
                else:
                    train_paths.append(test)

    print(f"Found {len(train_paths)} training and {len(test_paths)} test file paths in the dataset.")
    return train_paths, test_paths
                
    

def load_data_into_omen_dataset(
        n_sessions:int=-1,
        downsample_movements_factor:int=1,
        load_test_only=False,
        downsample_target_factor=1,
        ):

    train_paths, test_paths = collect_paths(load_test_only)
    print(f"Found {len(train_paths)} training and {len(test_paths)} test sessions in the dataset.")

    dt = int(1000/200) * downsample_target_factor
    variables = [DatasetVariable(f'target_{i}', 'hand', False) for i in range(n_outputs)]
    omen_ds =  Dataset('hand', '', -1, dt, 1, variables, [])
    session = DatasetSession('session', omen_ds.name, dt, variables, [], [], [])

    t = 0
    for i, path in enumerate(train_paths):
        if n_sessions > 0 and i >= n_sessions:
            break
        trials, t = numpy_fles_to_trials(path, downsample_target_factor, dt, t)
        session.train_trials.extend(trials)

    for i, path in enumerate(test_paths):
        if n_sessions > 0 and i >= n_sessions:
            break
        trials, t = numpy_fles_to_trials(path, downsample_target_factor, dt, t)
        session.test_trials.extend(trials)

    print(f"Amount of data: {len(session.train_trials)} train and {len(session.test_trials)} test trials")

    omen_ds.sessions.append(session)
    return omen_ds, session
