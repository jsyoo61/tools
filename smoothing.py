import statsmodels.api as sm
import numpy as np


def LPF(x, cutoff=0.5):
    '''Low Pass Filter. Eliminates high frequency components of wave x.

    Parameters
    ----------
    x: 1d-array

    cutoff: 0 ~ 1 (float)
    '''
    assert (0<=cutoff) and (cutoff <= 1), 'Cutoff frequency invalid. It must be within range 0~1'
    N = len(x)
    N_cutoff = round(cutoff * N/2)
    x_hat = np.fft.fft(x)
    x_hat[N_cutoff+1:-N_cutoff] = 0
    x = np.real(np.fft.ifft(x_hat))
    return x


def smooth_lowess(bis_history, frac=0.01):
    valid_bis_indices = np.where(bis_history > 0)[0]
    valid_bis_values = bis_history[valid_bis_indices].tolist()

    lowess_bis_history = bis_history.copy()
    lowess_bis_history[valid_bis_indices] = sm.nonparametric.lowess(
        [valid_bis_values[0]] * 30 + valid_bis_values + [valid_bis_values[-1]] * 30,
        np.arange(-30, len(valid_bis_values) + 30), frac=frac
    )[30:-30, 1]

    return lowess_bis_history


def smooth_LPF(bis_history, cutoff=0.03):
    valid_bis_indices = np.where(bis_history > 0)[0]
    valid_bis_values = bis_history[valid_bis_indices].tolist()

    lpf_bis_history = bis_history.copy()
    lpf_bis_history[valid_bis_indices] = LPF(np.array(valid_bis_values) / 98., cutoff) * 98

    return lpf_bis_history
