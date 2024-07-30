import sys
from loguru import logger

logger.remove()
logger.add(sys.stderr, level=1);

from utils.omen_utils import load_data_into_omen_dataset
from omen import OMEN
from abcidatasets.error_metrics import mse


print('\n\n')
omen_ds = load_data_into_omen_dataset(1)


omen_config = dict(
    activation= 'leaky_relu',
    beta= 2.5,
    embedding_dim= 64,
    head_n_layers= 1,
    input_sigma= 0.25,
    kernel_size= 5,
    lr= 0.0075,
    n_epochs= 250,
    n_hidden= 64,
    n_kernels= 1,
    n_layers= 2,
    sigma= 0.25,
)


betas = [4, 8, 8, 4, 4, 4, 4, 4]
sigmas = [1, 1, 1, 1, 0.5, 1, 1, 0.5]

# create and fit
omen = OMEN.from_config(omen_config)
logger.debug("Fitting OMEN")
omen.fit_session(
    omen_ds.sessions[0], plot_history=True, verbose=True, should_refine=False
)
logger.debug("Saving model")
omen.save(omen_ds.sessions[0].name, "models")

# evaluate
logger.debug("Predicting")
omen.predict_session(omen_ds.sessions[0], trial_set='train')

logger.debug("Calculating error")
err = mse(omen_ds.sessions[0], trial_set='train')
print(f"Training error: {err}")

print('\n\n')