import numpy as np

import librosa
from librosa.util import normalize


def mix(x0, x1, snr):
    """Mix two signals

    Args:
        x0 (numpy.ndarray): signal (n_samples,)
        x1 (numpy.ndarray): signal (n_samples,)
        snr (float): mixing coefficient applied on x1 (dB)

    Returns:
        numpy.ndarray: mixed signal (n_samples,)
    """
    # apply
    x0 = _norm_n_weight(x0, 0)  # set this signal as `signal`
    x1 = _norm_n_weight(x1, -snr)  # treat this as `noise`
    y = normalize(x0 + x1)
    return y


def _norm_n_weight(x, dB):
    """Normlize and weight to given dB ratio

    Args:
        x (numpy.ndarray): signal (n_samples,)
        dB (float): target dB (*ratio)

    Returns:
        numpy.ndarray: processed signal (n_samples,)
    """
    # normalize both signal
    x = normalize(x)

    # get the RMS of each signal
    rms = np.linalg.norm(x) / np.sqrt(len(x))

    # get the weight
    ratio = (10**(dB / 20)) / rms

    return x * ratio
