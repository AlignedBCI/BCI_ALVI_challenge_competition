import sys
from loguru import logger

logger.remove()
logger.add(sys.stderr, level=1);

import yaml

from utils.omen_utils import load_data_into_omen_dataset
from omen.hyperparameters.sweep import Sweeper


# ---------------------------------------------------------------------------- #
#                                   GET DATA                                   #
# ---------------------------------------------------------------------------- #

print('\n\n')
omen_ds = load_data_into_omen_dataset(1)


# ---------------------------------------------------------------------------- #
#                                     SWEEP                                    #
# ---------------------------------------------------------------------------- #
sweeper = Sweeper(
            'OMEN',
            omen_ds,
            'best_hps.yaml',
            '',
            num_cpus=48,
            num_gpus=4,
            n_sessions=1,
            verbose=True,
            num_gpus_per_job=0.5,
            num_cpus_per_job=6,
            multi_session=False,
        )
best_hps = sweeper.sweep()


# save to yaml
with open('best_hps.yaml', 'w') as fl:
    yaml.dump(best_hps, fl)
