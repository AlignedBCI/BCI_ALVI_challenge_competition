import sys
from loguru import logger
from rich.traceback import install

install(show_locals=False)
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
N_sessions = 10
omen_ds, _ = load_data_into_omen_dataset(
        N_sessions, 
        downsample_movements_factor=4, 
        load_test_only=False
)


# ---------------------------------------------------------------------------- #
#                                     SWEEP                                    #
# ---------------------------------------------------------------------------- #

space = {
        "n_hidden": tune.choice([32, 64, 128, 256, 512]),
        "n_layers": tune.choice([-1, 0, 1, 2,]),
        "embedding_dim": tune.choice([32, 64, 128]),
        "activation": tune.choice(["relu", "tanh", "leaky_relu"]),
        "input_sigma": tune.choice([0.1, 0.25,]),
        "kernel_size": tune.choice([1, 5, 11,]),
        "head_n_layers": tune.choice([-1, 1, 2]),
        "lr": tune.choice([1e-4, 1e-3, 1e-2]),
        "n_epochs": tune.choice([500, 1500, 2500, 5000, 10_000]),
        "beta": tune.choice([0.5, 2.5, 5.0, 8.0]),
        "sigma": tune.choice([0.2, 0.3, 0.4, 0.5, 0.6]),
        "n_kernels": tune.choice([1, 2, 4]),
}


sweeper = Sweeper(
            'OMEN',
            omen_ds,
            'best_hps.yaml',
            '',
            num_cpus=48,
            num_gpus=4,  # 6
            n_sessions=N_sessions,
            verbose=True,
            num_gpus_per_job=1.0,
            num_cpus_per_job=12,
            multi_session=False,
            space=space,
            seeds=[],
        #     n_samples=4,
        )
best_hps = sweeper.sweep()


# save to yaml
with open('best_hps.yaml', 'w') as fl:
    yaml.dump(best_hps, fl)
