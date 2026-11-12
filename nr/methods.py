"""Enhancement methods. Each maps (mono waveform, sr, Config) -> same-length waveform.

basic_filter          Butterworth HP -> LP -> band-stop (course baseline, Config.basic_stages)
lowpass / highpass / bandstop   each basic-filter stage alone (course notebook variants)
spectral_subtraction  Boll (1979) power subtraction (course baseline)
wiener_filter         Wiener gain + decision-directed a-priori SNR (course baseline)
mmse_lsa              Ephraim & Malah (1985) log-spectral amplitude estimator
omlsa                 Cohen & Berdugo (2001) OM-LSA on a tracked noise PSD (SPP), optional TSNR
"""
from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d
from scipy.special import exp1

from . import noise as noise_est
from .config import DEFAULT, Config
from .stft import istft, stft

EPS = 1e-12


# --------------------------------------------------------------------------
# Course baselines
# --------------------------------------------------------------------------
def butterworth_stages(x, sr, cfg: Config, stages) -> np.ndarray:
    """Butterworth SOS stages, forward/backward (zero phase, offline).

    Stages whose cut-off is at or above Nyquist are skipped (other sample rates).
    """
    y = np.asarray(x, dtype=np.float64).copy()
    if y.size < 2:  # nothing to filter
        return y
    nyq = sr / 2
    cutoffs = {"highpass": cfg.highpass_hz, "lowpass": cfg.lowpass_hz,
               "bandstop": cfg.bandstop_hz}
    for btype in stages:
        wn = cutoffs[btype]
        if np.max(wn) >= nyq:
            continue
        sos = signal.butter(cfg.filter_order, wn, btype=btype, fs=sr, output="sos")
        # the usual padding, bounded for very short inputs
        ntaps = 2 * len(sos) + 1 - min((sos[:, 2] == 0).sum(), (sos[:, 5] == 0).sum())
        y = signal.sosfiltfilt(sos, y, padlen=min(3 * int(ntaps), len(y) - 1))
    return y


def basic_filter(x, sr, cfg: Config):
    return butterworth_stages(x, sr, cfg, cfg.basic_stages)


def lowpass(x, sr, cfg: Config):
    return butterworth_stages(x, sr, cfg, ("lowpass",))


def highpass(x, sr, cfg: Config):
    return butterworth_stages(x, sr, cfg, ("highpass",))


def bandstop(x, sr, cfg: Config):
    return butterworth_stages(x, sr, cfg, ("bandstop",))


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
    noise = cfg.noise_scale * noise_est.lowest_energy_frames(z, t, sr, len(x), cfg)
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


def omlsa_gain_stft(z, noise, cfg: Config):
    """Apply the OM-LSA gain to a noisy STFT z for a (freq_bins, frames) noise PSD."""
    power = np.abs(z) ** 2
    gamma = np.minimum(power / noise, 1e4)
    xi_min = 10 ** (cfg.lsa_xi_min_db / 10)
    q_ratio = cfg.omlsa_q / (1 - cfg.omlsa_q)
    log_gmin = np.log(max(cfg.omlsa_gain_floor, 1e-6))
    out = np.empty_like(z)
    prev = np.zeros(power.shape[0])  # |G_H1 * Y|^2 of the previous frame
    for k in range(z.shape[1]):
        g_k = gamma[:, k]
        inst = np.maximum(g_k - 1.0, 0.0)
        xi = inst if k == 0 else cfg.dd_alpha * prev / noise[:, k] + (1 - cfg.dd_alpha) * inst
        xi = np.maximum(xi, xi_min)
        g1 = np.minimum(lsa_gain(xi, g_k), 1.0)
        if cfg.omlsa_tsnr:
            xi = np.maximum(g1 ** 2 * g_k, xi_min)
            g1 = np.minimum(lsa_gain(xi, g_k), 1.0)
        prev = g1 ** 2 * power[:, k]
        xi_s = uniform_filter1d(xi, size=cfg.omlsa_smooth_bins, mode="nearest")
        ga_s = uniform_filter1d(g_k, size=cfg.omlsa_smooth_bins, mode="nearest")
        v = np.minimum(xi_s * ga_s / (1.0 + xi_s), 200.0)
        p = 1.0 / (1.0 + q_ratio * (1.0 + xi_s) * np.exp(-v))
        g = np.exp(p * np.log(np.maximum(g1, 1e-6)) + (1 - p) * log_gmin)
        out[:, k] = g * z[:, k]
    return out


def omlsa(x, sr, cfg: Config):
    """Optimally-modified LSA (Cohen & Berdugo 2001) on a time-varying noise PSD.

    noise PSD   cfg.noise_estimator (nr/noise.py; "spp": Gerkmann & Hendriks 2012),
                estimated from the noisy input only
    a-priori    decision-directed on the speech-present estimate G_H1*Y,
                optionally refined in a second step (TSNR, Plapous et al. 2006)
    presence    p = 1 / (1 + q/(1-q) * (1+xi) * exp(-v)), xi and gamma smoothed
                over frequency, fixed a-priori absence probability q
    gain        G = G_H1^p * G_min^(1-p)
    """
    t, z = stft(x, sr, cfg)
    noise = noise_est.estimate(z, t, sr, len(x), cfg)
    return istft(omlsa_gain_stft(z, noise, cfg), sr, len(x), cfg)


# --------------------------------------------------------------------------
METHODS = {
    "basic_filter": basic_filter,
    "lowpass": lowpass,
    "highpass": highpass,
    "bandstop": bandstop,
    "spectral_subtraction": spectral_subtraction,
    "wiener_filter": wiener_filter,
    "mmse_lsa": mmse_lsa,
    "omlsa": omlsa,
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
