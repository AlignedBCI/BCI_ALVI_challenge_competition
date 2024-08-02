import sys
from loguru import logger
import matplotlib.pyplot as plt
from rich.traceback import install
import numpy as np
from sklearn.metrics import mean_squared_error
from rich.progress import track
from rich import print

install(show_locals=False)

logger.remove()
logger.add(sys.stderr, level=1);
logger.add(sys.stdout, level=1);

from utils.omen_utils import load_data_into_omen_dataset, load_submission_dataset, reshape_and_average, emg_amplitude
from omen import OMEN

# ---------------------------------------------------------------------------- #
#                                  GET DATA                                    #
# ---------------------------------------------------------------------------- #
print('\n\n')
omen_ds_train, session = load_data_into_omen_dataset(
    5, 
    downsample_movements_factor=1, 
    load_test_only=True
)


sub_X, sub_Y = load_submission_dataset()   # 72 mvmts, X: 214380 x * - Y: 26829 x 20

f, ax = plt.subplots(5, 1, figsize=(12, 10))
for i in range(5):
    ax[i].plot(sub_Y.values[:, i])
    ax[i].set(xlim=[0, 5000])
plt.savefig(f'omen_GT.png')


# ---------------------------------------------------------------------------- #
#                                   GET OMEN                                   #
# ---------------------------------------------------------------------------- #
"""
0.118

GT      0.1837
train   0.0404
test    0.12
"""
# omen_config = {
#         'n_hidden': 64, 
#         'n_layers': -1, 
#         'embedding_dim': 128, 
#         'activation':  'leaky_relu', 'input_sigma': 0.25, 
#         'kernel_size': 11, 'head_n_layers': 2, 
#         'lr': 0.005, 'n_epochs': 5000, 
#         'beta': 4.0, 
#         'sigma': 0.3, 
#         'n_kernels': 4, 
#         'N': 48, 
#         'patience': 250,
# }

omen_config = {'n_hidden': 512, 'n_layers': -1, 'embedding_dim': 32, 'activation': 'relu', 
                'input_sigma': 0.25, 'kernel_size': 11, 'head_n_layers': 2, 'lr': 0.0001, 'n_epochs': 10000, 
                'beta': 1.5, 'sigma': 0.25, 'n_kernels': 4, 'N': 48, 'patience': 1000
}



# omen_config = {
#    'n_hidden': 32,
#    'n_layers': -1,
#    'embedding_dim': 128,
#    'activation': 'leaky_relu',
#    'input_sigma': 0.25,
#    'kernel_size': 1,
#    'head_n_layers': 1,
#    'lr': 0.001,
#    'n_epochs': 1500,
#    'beta': 0.5,
#    'sigma': 0.6,
#    'n_kernels': 2
# }

# create and fit
omen = OMEN.from_config(omen_config)

# ---------------------------------------------------------------------------- #
#                                 FIT TO TRAIN                                 #
# ---------------------------------------------------------------------------- #
print("Fitting OMEN")
omen.fit_session(
    session, plot_history=True, verbose=True, should_refine=False
)

print(omen)
omen.save('omen', "models")

# ---------------------------------------------------------------------------- #
#                                     EVAL                                     #
# ---------------------------------------------------------------------------- #
sd = omen.lut[session.name]
predictions = []
for X in track(sub_X, description='Predicting GT'):
        X = X.T  # n channels by n samples
        X = reshape_and_average(emg_amplitude(X), 8) 
        pred = omen.predict(
            sd.on_predict_start(
                            X.T, do_ue=False
                        )
        )  
        predictions.append(pred)

Yhat = np.concatenate(predictions)
if Yhat.shape != sub_Y.shape:
     print(f"[bold red]Invalid prediction shape: {Yhat.shape}. It should be {sub_Y.shape}")


f, ax = plt.subplots(5, 1, figsize=(12, 10))
for i in range(5):
    ax[i].plot(sub_Y.values[:, i], lw=2, color='k')
    ax[i].plot(Yhat[:, i], color='red', alpha=.5)
    ax[i].set(xlim=[0, 2_000])

plt.savefig(f'omen_GT_preds.png')
T = Yhat.shape[0]
err = mean_squared_error(sub_Y.values[:T, :], Yhat)
print(f"[bold green]Submission error: {err:.4f}[/]\n\n")


print("Predicting")
for tset in ('train', 'test'):
    omen.predict_session(session, trial_set=tset, verbose=True)
    
    Y = np.concatenate([t.Y for t in session.get_trials_subset(tset)])
    Yhat = np.concatenate([t.Ypred for t in session.get_trials_subset(tset)])
    err = mean_squared_error(Y, Yhat)
    print(f"[orange]{tset} error: {err:.4f}[/]")



# ---------------------------------------------------------------------------- #
#                                     PLOT                                     #
# ---------------------------------------------------------------------------- #
n_trials_to_plot=10
n_ch_to_plot=5

print("Final plot")
for tset in ('train', 'test'):
    f, axes = plt.subplots(n_ch_to_plot, 1, figsize=(15, 6), sharex=True)

    trials = session.get_trials_subset(tset)
    sorted_trials = sorted(trials, key=lambda x: x.time[0])

    for trial in sorted_trials[:n_trials_to_plot]:
        sd = omen.lut[session.name]
        trial.Ypred = omen.predict(
            sd.on_predict_start(
                            trial.X, do_ue=False
                        )
        )

        for i in range(n_ch_to_plot):
            axes[i].plot(trial.time, trial.Y[:, i], label='true', color='k', lw=2)
            axes[i].plot(trial.time, trial.Ypred[:, i], label='pred', color='r', lw=1)

    # save figure as png
    plt.savefig(f'omen_{tset}.png')

    print('\n\n')