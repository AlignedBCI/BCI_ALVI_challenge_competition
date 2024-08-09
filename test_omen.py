import sys
from loguru import logger
import matplotlib.pyplot as plt
from rich.traceback import install
import numpy as np
from sklearn.metrics import mean_squared_error
from rich.progress import track
from rich import print
from rich.pretty import pprint
import torch
install(show_locals=False)

logger.remove()
logger.add(sys.stderr, level=1);
logger.add(sys.stdout, level=1);

from utils.omen_utils import load_data_into_omen_dataset, load_submission_dataset, emg_amplitude, recombine_predictions
from omen import OMEN
from omen.augmentations import WaveletNoiseInjection, MagnitudeWarping

KAGGLE_BEST = 0.11711
test_session_name = 'fedya_tropin_standart_elbow_left'

# ---------------------------------------------------------------------------- #
#                                  GET DATA                                    #
# ---------------------------------------------------------------------------- #
print('\n\n')
downsample_target_factor = 8
n_sessions = 4

omen_ds = load_data_into_omen_dataset(
    n_sessions, 
    downsample_movements_factor=-1, 
    downsample_target_factor=downsample_target_factor,
    load_test_only=True
)

sub_X, sub_Y = load_submission_dataset()   # 72 mvmts, X: 214380 x * - Y: 26829 x 20

# TODO GPU memory efficient training with multiple sessions

# ---------------------------------------------------------------------------- #
#                                   GET OMEN                                   #
# ---------------------------------------------------------------------------- #
"""


When trained with all 22 train sessions
GT:     0.1254 --- +7.09%
Train:  0.0947
Test:   0.1150

When trained with 8 train sesssions
GT:     0.1171  --- +4.2%
Train:
Test:

When trained on only the test session
GT:     0.1453  --- +24.07%
Train:  0.0711
Test:   0.1084

"""

# omen_config = {'n_hidden': 512, 'n_layers': -1, 'embedding_dim': 32, 'activation': 'relu', 
#                 'input_sigma': 0.25, 'kernel_size': 11, 'head_n_layers': 2, 'lr': 0.0005, 'n_epochs': 5000, 
#                 'beta': .5, 'sigma': 0.5, 'n_kernels': 4, 'patience': 50,
# }

omen_config = {'n_hidden': 512, 'n_layers': -1, 'embedding_dim': 32, 'activation': 'relu', 
                'input_sigma': 0.25, 'kernel_size': 11, 'head_n_layers': 2, 'lr': 0.0005, 'n_epochs': 5000, 
                'beta': 10,  # ! changed 
                'sigma': 0.5, 'n_kernels': 4, 'patience': 50,
}



# create and fit
omen = OMEN.from_config(omen_config)

# ---------------------------------------------------------------------------- #
#                                 FIT TO TRAIN                                 #
# ---------------------------------------------------------------------------- #
print("Fitting OMEN")
omen.fit_session(
    omen_ds.sessions[0], 
    plot_history=False, 
    verbose=True, 
    should_refine=False, 
    augmentations=[WaveletNoiseInjection(), MagnitudeWarping()]
)
# omen.fit_sessions(omen_ds.sessions, plot_history=False, verbose=True, should_refine=False)

# omen.refine_cans(verbose=True, sess_to_refine=[test_session_name])

print(omen)
omen.save('omen', "models")

# ---------------------------------------------------------------------------- #
#                                     EVAL                                     #
# ---------------------------------------------------------------------------- #
omen.sd = sd = omen.lut[test_session_name]

# ------------------------------- ground truth ------------------------------- #
predictions = []
for X in track(sub_X, description='Predicting GT'):
        X = X.T  # n channels by n samples
        X = emg_amplitude(X)[:, ::downsample_target_factor]
        pred = omen.predict(
            sd.on_predict_start(
                            X.T, do_ue=False
                        )
        )  

        # make sure overall downsampling is 8x
        if downsample_target_factor < 8:
            new_factor = 8 // downsample_target_factor
            pred = pred[::new_factor, :]

        # sum frequency components
        pred = recombine_predictions(pred)
             
        gt_len = X.shape[1]
        pred = pred[:gt_len, :]
        predictions.append(pred)

Yhat = np.concatenate(predictions)
if Yhat.shape != sub_Y.shape:
     print(f"[bold red]Invalid prediction shape: {Yhat.shape}. It should be {sub_Y.shape}")


f, ax = plt.subplots(5, 1, figsize=(20, 18))
for i in range(5):
    ax[i].plot(sub_Y.values[:, i], lw=2, color='k')
    ax[i].plot(Yhat[:, i], color='red', alpha=.5)
    ax[i].set(xlim=[0, 12000])

plt.savefig(f'omen_GT_preds_{n_sessions}.png')
err = mean_squared_error(sub_Y.values, Yhat)
color = 'green' if err < KAGGLE_BEST else 'red'
change = (err - KAGGLE_BEST) / KAGGLE_BEST * 100
print(f"[bold {color}]Submission error: {err:.4f}[/] -- Kaggle best {KAGGLE_BEST}. Change [{color}]{change:.2f}%[/]\n\n")



# ---------------------------------- session --------------------------------- #
print("Predicting")
for tset in ('train', 'test',):
    omen.predict_session(omen_ds.sessions[0], trial_set=tset, verbose=True)
    
    Y = np.concatenate([t.Y for t in omen_ds.sessions[0].get_trials_subset(tset)])
    Yhat = np.concatenate([t.Ypred for t in omen_ds.sessions[0].get_trials_subset(tset)])
    err = mean_squared_error(Y, Yhat)
    print(f"[orange]{tset} error: {err:.4f}[/]")
# session.visualize()
# plt.savefig("session.png")


# ---------------------------------------------------------------------------- #
#                                     PLOT                                     #
# ---------------------------------------------------------------------------- #
n_trials_to_plot=25
n_ch_to_plot=12

print("Final plot")
for tset in ('train', 'test'):
    f, axes = plt.subplots(n_ch_to_plot, 1, figsize=(15, 20), sharex=True)

    trials = omen_ds.sessions[0].get_trials_subset(tset)
    sorted_trials = sorted(trials, key=lambda x: x.time[0])

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
    plt.savefig(f'omen_{tset}_nsess_{n_sessions}.png')

print('\n\n')
pprint(omen_config)