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
from abcidatasets.dataset.utils import make_acausal_kernel

DATA_PATH = "./dataset_v2_blocks"
n_inputs, n_outputs = 8, 20


def emg_amplitude(x):
    kernel = make_acausal_kernel(9)
    amp = np.zeros_like(x)

    for i in range(x.shape[0]):
        amp[i] = np.convolve(np.abs(x[i]), kernel, mode='same')
    return amp

def reshape_and_average(x, n):
    """
        x is emg data of shape n_inputs x T. 
        The function should return an array of shape n_inputs x T//N
        where each value is the average of the corresponding N values in x
    """
    T = x.shape[1]
    x = x[:, :T//n*n]
    return x.reshape(n_inputs, -1, n).mean(axis=2)
    

def ds_to_session(name, myo_session_data, omen_ds, train_config, dt, variables, downsample_movements_factor):
    logger.debug(f"Creating session {name}")
    movements = [xy for xy in myo_session_data][::downsample_movements_factor]
    train_Xs = np.concatenate([
        reshape_and_average(emg_amplitude(x), train_config.down_sample_target) 
        # x[:, ::train_config.down_sample_target]
        for x, _ in movements
    ], axis=1)
    train_Ys = np.concatenate([y for _, y in movements], axis=1)

    data = {}
    for i in range(n_inputs):
        data[f"myo_{i}"] = train_Xs[i]
    for i in range(n_outputs):
        data[f"target_{i}"] = train_Ys[i]
    data = pd.DataFrame(data)
    data.reset_index(drop=True, inplace=True)
    data['time'] = np.arange(data.shape[0]) * dt


    sess = omen_ds.session_from_df(
        data, 'hand', name, variables, dt, unit_prefix='myo_', trial_duration_seconds=5
    )
    sess.train_trials = sess.all_trials  # we're only using the training set to create the session so this is ok
    omen_ds.sessions.append(sess)
    return sess

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



def load_data_into_omen_dataset(
        n_sessions:int=-1,
        downsample_movements_factor:int=1,
        load_test_only=False,
        ):

    if load_test_only:
        data_paths = dict(
            datasets=[DATA_PATH],
            hand_type = ['left', 'right'], # [left, 'right']
            human_type = ['amputant'], # [amputant, 'health']
            test_dataset_list = ['fedya_tropin_standart_elbow_left'],  # don't change this !
            random_sampling=False,
        )
    else:
        data_paths = dict(
            datasets=[DATA_PATH],
            hand_type = ['left', 'right'], # [left, 'right']
            human_type = ['health', 'amputant'], # [amputant, 'health']
            # test_dataset_list = ['all'], # ['fedya_tropin_standart_elbow_left'],  # don't change this !
            test_dataset_list = ['fedya_tropin_standart_elbow_left'],  # don't change this !
            random_sampling=False,
        )

    # define a config object to keep track of data variables
    data_config = creating_dataset.DataConfig(**data_paths)

    train_paths, val_paths = creating_dataset.get_train_val_pathes(data_config)
    print(f"Found {len(train_paths)} training and {len(val_paths)} test sessions in the dataset.")

    train_config = replace(data_config, samples_per_epoch=int(data_config.samples_per_epoch / len(train_paths)))               
    val_config = replace(data_config, random_sampling=False, samples_per_epoch=None)

    logger.debug(f"Found {len(train_paths)} training paths and {len(val_paths)} validation paths")

    dt = int(1000/200) * train_config.down_sample_target
    variables = [DatasetVariable(f'target_{i}', 'hand', False) for i in range(n_outputs)]
    omen_ds_train =  Dataset('hand', '', -1, dt, 1, variables, [])
    omen_ds_test =  Dataset('hand', '', -1, dt, 1, variables, [])

    for omen_ds, paths in zip((omen_ds_train, omen_ds_test), (train_paths, val_paths)):
        for i, path in track(enumerate(paths), description="Loading data", total=len(paths)):
            if n_sessions  > 0 and i >= n_sessions:
                break
            name = path.parent.parent.name
            ds = init_dataset(train_config, path)
            sess = ds_to_session(name, ds, omen_ds, train_config, dt, variables, downsample_movements_factor)
            logger.debug(f"Added session {name} with {sess.n_train_trials} trials and {sess.X_train.shape[0]} samples ({sess.X_train.shape[0]/25:.2f} seconds)")
            

    print('Collecting all data in a single session')
    master_session = DatasetSession(
        'master_session', omen_ds_train.name, omen_ds_train.delta_t, 
        omen_ds_train.variables, [], [], []
    )
    for tset, ds in zip(('train', 'test'), (omen_ds_train, omen_ds_test)):
        for sess in ds.sessions:
            if tset == 'train':
                master_session.train_trials.extend(sess.train_trials)
            else:
                master_session.test_trials.extend(sess.train_trials)


    omen_ds_train.sessions = [master_session]
    print(f"Master session has {master_session.n_train_trials} train and {master_session.n_test_trials} test trials")

    # we need to cleanup the trials times
    for tset in ('train', 'test'):
        t = 0
        for trial in master_session.get_trials_subset(tset):
            trial.time = (trial.time - trial.time[0]) + t
            t = trial.time[-1]

    return omen_ds_train, master_session
