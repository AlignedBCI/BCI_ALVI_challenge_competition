import sys
from loguru import logger

logger.remove()
logger.add(sys.stderr, level=1);

import yaml
from ray import tune
from utils.omen_utils import load_data_into_omen_dataset

import omen
omen.hyperparameters.sweep.N_ITERATIONS = 2

from omen.hyperparameters.sweep import Sweeper


# ---------------------------------------------------------------------------- #
#                                   GET DATA                                   #
# ---------------------------------------------------------------------------- #

print('\n\n')
omen_ds = load_data_into_omen_dataset(1, downsample_movements_factor=6)


# ---------------------------------------------------------------------------- #
#                                     SWEEP                                    #
# ---------------------------------------------------------------------------- #

space = {
        "n_hidden": tune.choice([32, 64, 128, 256]),
        "n_layers": tune.choice([-1, 0, 1, 2,]),
        "embedding_dim": tune.choice([32, 64, 128]),
        "activation": tune.choice(["relu", "tanh", "leaky_relu"]),
        "input_sigma": tune.choice([0.1, 0.25,]),
        "kernel_size": tune.choice([1, 5, 11,]),
        "head_n_layers": tune.choice([-1, 1, 2]),
        "lr": tune.choice([1e-4, 1e-3, 1e-2]),
        "n_epochs": tune.choice([500, 1500,]),
        "beta": tune.choice([0.5, 2.5, 5.0]),
        "sigma": tune.quniform(0.05, 0.6, 0.05),
        "n_kernels": tune.choice([1, 2, 4]),
}


sweeper = Sweeper(
            'OMEN',
            omen_ds,
            'best_hps.yaml',
            '',
            num_cpus=48,
            num_gpus=4,
            n_sessions=1,
            verbose=True,
            num_gpus_per_job=0.25,
            num_cpus_per_job=4,
            multi_session=False,
            space=space,
            seeds=[]
        )
best_hps = sweeper.sweep()


# save to yaml
with open('best_hps.yaml', 'w') as fl:
    yaml.dump(best_hps, fl)
