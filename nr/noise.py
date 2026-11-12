"""Noise power spectral density (PSD) estimators. Only the noisy input is used."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d

from .config import Config
from .stft import frame_params

EPS = 1e-12


def lowest_energy_frames(z, t, sr, length, cfg: Config) -> np.ndarray:
    """Course baseline: mean power of the lowest-energy frames, fixed per file.

    Returns a (freq_bins,) PSD. Cannot follow noise that changes over time.
    """
    n_fft, _ = frame_params(sr, cfg)
    power = np.abs(z) ** 2
    centers = np.rint(t * sr).astype(int)
    full = np.flatnonzero((centers >= n_fft // 2) & (centers + n_fft // 2 <= length))
    if not len(full):
        full = np.arange(power.shape[1])
    energy = power[:, full].mean(axis=0)
    count = max(1, int(np.ceil(cfg.noise_fraction * len(full))))
    selected = full[np.argsort(energy, kind="stable")[:count]]
    est = power[:, selected].mean(axis=1)
    return np.maximum(uniform_filter1d(est, size=3, mode="nearest"), EPS)
