import numpy as np
from dataclasses import replace
import pandas as pd
from loguru import logger

from .creating_dataset import init_dataset
from .augmentations import get_default_transform
from . import creating_dataset

from abcidatasets import Dataset, DatasetVariable
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
    

def ds_to_session(name, myo_session_data, omen_ds, train_config, dt, variables):
    logger.debug(f"Creating session {name}")
    train_Xs = np.concatenate([
        reshape_and_average(emg_amplitude(x), train_config.down_sample_target) 
        # x[:, ::train_config.down_sample_target]
        for x, _ in myo_session_data
    ], axis=1)
    train_Ys = np.concatenate([y for _, y in myo_session_data], axis=1)

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


def load_data_into_omen_dataset(n_sessions:int=-1):

    data_paths = dict(
        datasets=[DATA_PATH],
        hand_type = ['left', 'right'], # [left, 'right']
        human_type = ['health'], # 'amputant'], # [amputant, 'health']
        test_dataset_list = ['fedya_tropin_standart_elbow_left'],  # don't change this !
        random_sampling=False,
    )

    # get transforms
    p_transform = 0.1  # probability of applying the transform
    transform = get_default_transform(p_transform)

    # define a config object to keep track of data variables
    data_config = creating_dataset.DataConfig(**data_paths)

    train_paths, val_paths = creating_dataset.get_train_val_pathes(data_config)


    train_config = replace(data_config, samples_per_epoch=int(data_config.samples_per_epoch / len(train_paths)))               
    val_config = replace(data_config, random_sampling=False, samples_per_epoch=None)

    logger.debug(f"Found {len(train_paths)} training paths and {len(val_paths)} validation paths")

    dt = int(1000/200) * train_config.down_sample_target
    variables = [DatasetVariable(f'target_{i}', 'hand', False) for i in range(n_outputs)]
    omen_ds =  Dataset('hand', '', -1, dt, 1, variables, [])


    for i, path in enumerate(train_paths):
        if n_sessions  > 0 and i >= n_sessions:
            break
        name = path.parent.parent.name
        ds = init_dataset(train_config, path, transform=transform)
        sess = ds_to_session(name, ds, omen_ds, train_config, dt, variables)
        logger.debug(f"Added session {name} with {sess.n_train_trials} trials")
        

    return omen_ds
