"""Enhancement methods. Each maps (mono waveform, sr, Config) -> same-length waveform.

basic_filter          Butterworth HP -> LP -> band-stop (course baseline)
spectral_subtraction  Boll (1979) power subtraction (course baseline)
wiener_filter         Wiener gain + decision-directed a-priori SNR (course baseline)
mmse_lsa              Ephraim & Malah (1985) log-spectral amplitude estimator
"""
from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.special import exp1

from . import noise as noise_est
from .config import DEFAULT, Config
from .stft import istft, stft

EPS = 1e-12


# --------------------------------------------------------------------------
# Course baselines
# --------------------------------------------------------------------------
def basic_filter(x, sr, cfg: Config):
    y = np.asarray(x, dtype=np.float64).copy()
    nyq = sr / 2
    stages = []
    if cfg.highpass_hz < nyq:
        stages.append(("highpass", cfg.highpass_hz))
    if cfg.lowpass_hz < nyq:
        stages.append(("lowpass", cfg.lowpass_hz))
    if cfg.bandstop_hz[1] < nyq:
        stages.append(("bandstop", cfg.bandstop_hz))
    for btype, wn in stages:
        sos = signal.butter(cfg.filter_order, wn, btype=btype, fs=sr, output="sos")
        ntaps = 2 * len(sos) + 1 - min((sos[:, 2] == 0).sum(), (sos[:, 5] == 0).sum())
        y = signal.sosfiltfilt(sos, y, padlen=min(3 * int(ntaps), len(y) - 1))
    return y


def spectral_subtraction(x, sr, cfg: Config):
    t, z = stft(x, sr, cfg)
    power = np.abs(z) ** 2
    noise = noise_est.lowest_energy_frames(z, t, sr, len(x), cfg)[:, None]
    target = np.maximum(power - cfg.subtraction_alpha * noise, cfg.subtraction_floor * power)
    gain = np.sqrt(np.clip(target / np.maximum(power, EPS), 0, 1))
    return istft(gain * z, sr, len(x), cfg)


def wiener_filter(x, sr, cfg: Config):
    t, z = stft(x, sr, cfg)
    power = np.abs(z) ** 2
    noise = noise_est.lowest_energy_frames(z, t, sr, len(x), cfg)
    gamma = power / noise[:, None]
    out = np.empty_like(z)
    prev = np.zeros_like(noise)
    for k in range(z.shape[1]):
        inst = np.maximum(gamma[:, k] - 1.0, 0.0)
        xi = inst if k == 0 else cfg.dd_alpha * prev / noise + (1 - cfg.dd_alpha) * inst
        g = np.clip(xi / (1 + xi), cfg.wiener_gain_floor, 1)
        out[:, k] = g * z[:, k]
        prev = np.abs(out[:, k]) ** 2
    return istft(out, sr, len(x), cfg)


# --------------------------------------------------------------------------
# MMSE log-spectral amplitude estimator
# --------------------------------------------------------------------------
def lsa_gain(xi, gamma):
    """G = xi/(1+xi) * exp(0.5 * E1(v)),  v = xi*gamma/(1+xi)."""
    v = np.maximum(xi * gamma / (1.0 + xi), 1e-10)
    return xi / (1.0 + xi) * np.exp(0.5 * exp1(v))


def mmse_lsa(x, sr, cfg: Config):
    t, z = stft(x, sr, cfg)
    power = np.abs(z) ** 2
    noise = noise_est.lowest_energy_frames(z, t, sr, len(x), cfg)
    gamma = np.minimum(power / noise[:, None], 1e4)
    xi_min = 10 ** (cfg.lsa_xi_min_db / 10)
    out = np.empty_like(z)
    prev = np.zeros(power.shape[0])
    for k in range(z.shape[1]):
        inst = np.maximum(gamma[:, k] - 1.0, 0.0)
        xi = inst if k == 0 else cfg.dd_alpha * prev / noise + (1 - cfg.dd_alpha) * inst
        xi = np.maximum(xi, xi_min)
        g = np.clip(lsa_gain(xi, gamma[:, k]), cfg.lsa_gain_floor, 1.0)
        out[:, k] = g * z[:, k]
        prev = np.abs(out[:, k]) ** 2
    return istft(out, sr, len(x), cfg)


# --------------------------------------------------------------------------
METHODS = {
    "basic_filter": basic_filter,
    "spectral_subtraction": spectral_subtraction,
    "wiener_filter": wiener_filter,
    "mmse_lsa": mmse_lsa,
}


def enhance(x, sr: int, method: str, cfg: Config = DEFAULT) -> np.ndarray:
    """Enhance a mono float waveform; 'noisy_input' returns it unchanged."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("enhance() expects a mono waveform")
    if x.size == 0:
        return x.copy()
    if method == "noisy_input":
        return x.copy()
    y = METHODS[method](x, sr, cfg)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    if y.shape != x.shape:
        raise RuntimeError(f"{method} changed the length: {x.shape} -> {y.shape}")
    return y
