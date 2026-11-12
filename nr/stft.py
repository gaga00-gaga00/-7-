"""STFT / ISTFT with a Hann window (perfect reconstruction with unit gain)."""
from __future__ import annotations

import numpy as np
from scipy import signal

from .config import Config


def frame_params(sr: int, cfg: Config) -> tuple[int, int]:
    n_fft = int(round(cfg.win_sec * sr))
    n_fft += n_fft % 2
    hop = max(1, int(round(cfg.hop_sec * sr)))
    return n_fft, hop


def stft(x: np.ndarray, sr: int, cfg: Config):
    """Return (frame_times, Z) with Z of shape (freq_bins, frames)."""
    n_fft, hop = frame_params(sr, cfg)
    if x.size < n_fft:
        x = np.pad(x, (0, n_fft - x.size))
    _, t, z = signal.stft(x, fs=sr, window="hann", nperseg=n_fft,
                          noverlap=n_fft - hop, nfft=n_fft,
                          boundary="zeros", padded=True)
    return t, z


def istft(z: np.ndarray, sr: int, length: int, cfg: Config) -> np.ndarray:
    n_fft, hop = frame_params(sr, cfg)
    _, y = signal.istft(z, fs=sr, window="hann", nperseg=n_fft,
                        noverlap=n_fft - hop, nfft=n_fft,
                        input_onesided=True, boundary=True)
    if y.size < length:
        y = np.pad(y, (0, length - y.size))
    return y[:length]
