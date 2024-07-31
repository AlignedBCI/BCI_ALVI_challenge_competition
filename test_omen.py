import sys
from loguru import logger
import matplotlib.pyplot as plt

logger.remove()
logger.add(sys.stderr, level=1);
logger.add(sys.stdout, level=1);

from utils.omen_utils import load_data_into_omen_dataset
from omen import OMEN
from abcidatasets.error_metrics import mse

# ---------------------------------------------------------------------------- #
#                                  GET DATA                                    #
# ---------------------------------------------------------------------------- #
print('\n\n')
omen_ds_train, omen_ds_test = load_data_into_omen_dataset(4, downsample_movements_factor=4, collate_sessions=True)

# ---------------------------------------------------------------------------- #
#                                   GET OMEN                                   #
# ---------------------------------------------------------------------------- #
omen_config = {'n_hidden': 64, 'n_layers': 1, 
        'embedding_dim': 128, 'activation': 'leaky_relu', 'input_sigma': 0.25, 
        'kernel_size': 11, 'head_n_layers': 2, 'lr': 0.015, 'n_epochs': 5000, 
        'beta': 12.0, 'sigma': 0.55, 'n_kernels': 4, 'N': 48, 'patience': 500
}

# create and fit
omen = OMEN.from_config(omen_config)

# ---------------------------------------------------------------------------- #
#                                 FIT TO TRAIN                                 #
# ---------------------------------------------------------------------------- #
logger.debug("Fitting OMEN")
omen.fit_session(
    omen_ds_train.sessions[0], plot_history=True, verbose=True, should_refine=False
)
logger.debug("Saving model")
omen.save('omen', "models")

# ---------------------------------------------------------------------------- #
#                                     EVAL                                     #
# ---------------------------------------------------------------------------- #
# evaluate
logger.debug("Predicting")

for _type, ds in zip(('Train', 'Test'), (omen_ds_train, omen_ds_test)):
    omen.predict_session(ds.sessions[0], trial_set='train')
    err = mse(ds.sessions[0], trial_set='train')
    print(f"{_type} error: {err}")


# ---------------------------------------------------------------------------- #
#                                     PLOT                                     #
# ---------------------------------------------------------------------------- #
n_trials_to_plot=10
n_ch_to_plot=20

for _type, ds in zip(('Train', 'Test'), (omen_ds_train, omen_ds_test)):
    f, axes = plt.subplots(n_ch_to_plot, 1, figsize=(15, 6), sharex=True)

    starts = [trial.time[0] for trial in ds.sessions[0].train_trials]
    sorted_trials = [trial for _, trial in sorted(zip(starts, ds.sessions[0].train_trials))]

    for trial in sorted_trials[:n_trials_to_plot]:
        sd = omen.lut[omen_ds.sessions[0].name]
        trial.Ypred = omen.predict(
            sd.on_predict_start(
                            trial.X, do_ue=False
                        )
        )

        for i in range(n_ch_to_plot):
            axes[i].plot(trial.time, trial.Y[:, i], label='true', color='k', lw=2)
            axes[i].plot(trial.time, trial.Ypred[:, i], label='pred', color='r', lw=1)

    # save figure as png
    plt.savefig(f'omen_{_type}.png')

    print('\n\n')