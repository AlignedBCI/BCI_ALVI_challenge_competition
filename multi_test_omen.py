import sys
from loguru import logger
import matplotlib.pyplot as plt
from rich.traceback import install
import numpy as np
from sklearn.metrics import mean_squared_error
from rich.progress import track
from rich import print
from rich.pretty import pprint

KAGGLE_BEST = 0.11711


install(show_locals=False)
logger.remove()
logger.add(sys.stderr, level=1);
logger.add(sys.stdout, level=1);

from utils.omen_utils import load_data_into_omen_dataset, load_submission_dataset, reshape_and_average, emg_amplitude
from omen import OMEN
from abcidatasets.error_metrics import mse


print('\n\n')
omen_ds, _ = load_data_into_omen_dataset(-1, downsample_movements_factor=1, downsample_target_factor=8, group_sessions=False)
session = omen_ds.sessions[0]

test_session_name = 'fedya_tropin_standart_elbow_left_test'


omen_config = {'n_hidden': 256, 'n_layers': 2, 'embedding_dim': 128, 
        'activation': 'leaky_relu', 'input_sigma': 0.25, 'kernel_size': 11, 
        'head_n_layers': 2, 'lr': 0.005, 'n_epochs': 5500, 'beta': 10, 
        'sigma': 0.3, 'n_kernels': 1,  'patience': 100
}


# create and fit
omen = OMEN.from_config(omen_config)
logger.debug("Fitting OMEN")
omen.fit_sessions(
    omen_ds.sessions, plot_history=True, verbose=True, should_refine=False
)
logger.debug("Saving model")
omen.save("multi", "models")

# evaluate
for tset in ('test',):
    omen.predict_session(session, trial_set=tset, verbose=True)
    
    Y = np.concatenate([t.Y for t in session.get_trials_subset(tset)])
    Yhat = np.concatenate([t.Ypred for t in session.get_trials_subset(tset)])
    err = mean_squared_error(Y, Yhat)
    print(f"[orange]{tset} error: {err:.4f}[/]")



# ----------------------------- submission error ----------------------------- #
sub_X, sub_Y = load_submission_dataset()   # 72 mvmts, X: 214380 x * - Y: 26829 x 20

omen.sd = sd = omen.lut[test_session_name]
predictions = []
for X in track(sub_X, description='Predicting GT'):
        X = X.T  # n channels by n samples
        X = reshape_and_average(emg_amplitude(X), 8) 
        pred = omen.predict(
            sd.on_predict_start(
                            X.T, do_ue=True
                        )
        )
        predictions.append(pred)

Yhat = np.concatenate(predictions)



f, ax = plt.subplots(5, 1, figsize=(12, 10))
for i in range(5):
    ax[i].plot(sub_Y.values[:, i], lw=2, color='k')
    ax[i].plot(Yhat[:, i], color='red', alpha=.5)
    ax[i].set(xlim=[0, 2000])

plt.savefig(f'omen_multi_GT_preds.png')
T = Yhat.shape[0]
err = mean_squared_error(sub_Y.values[:T, :], Yhat)
color = 'green' if err < KAGGLE_BEST else 'red'
change = (err - KAGGLE_BEST) / KAGGLE_BEST * 100
print(f"[bold {color}]Submission error: {err:.4f}[/] -- Kaggle best {KAGGLE_BEST}. Change [{color}]{change:.2f}%[/]\n\n")

print('\n\n')