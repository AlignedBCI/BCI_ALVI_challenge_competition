import sys
from loguru import logger

logger.remove()
logger.add(sys.stderr, level=1);
logger.add(sys.stdout, level=1);

from utils.omen_utils import load_data_into_omen_dataset
from omen import OMEN
from abcidatasets.error_metrics import mse


print('\n\n')
omen_ds = load_data_into_omen_dataset(4, downsample_movements_factor=4)
session = omen_ds.sessions[0]

omen_config = {'n_hidden': 64, 'n_layers': 2, 'embedding_dim': 128, 
        'activation': 'leaky_relu', 'input_sigma': 0.25, 'kernel_size': 11, 
        'head_n_layers': 2, 'lr': 0.01, 'n_epochs': 1500, 'beta': 0.5, 
        'sigma': 0.55, 'n_kernels': 4, 'N': 48, 'patience': 500
}

betas = [4, 8, 8, 4, 4, 4, 4, 4]
sigmas = [1, 1, 1, 1, 0.5, 1, 1, 0.5]

# create and fit
omen = OMEN.from_config(omen_config)
logger.debug("Fitting OMEN")
omen.fit_sessions(
    omen_ds.sessions, plot_history=True, verbose=True, should_refine=True
)
logger.debug("Saving model")
omen.save("multi", "models")

# evaluate
logger.debug("Predicting")
omen.predict_session(session, trial_set='train')

logger.debug("Calculating error")
err = mse(omen_ds.sessions[0], trial_set='train')
print(f"Training error: {err}")

print('\n\n')