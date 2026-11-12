"""Noise power spectral density (PSD) estimators. Only the noisy input is used.

Every estimator returns a (freq_bins, frames) PSD so that the gain functions can
follow noise that changes over time. Processing is offline (whole file known),
so the trackers may look both ways in time.

lowest_energy  course baseline: mean of the lowest-energy frames, fixed per file
spp            MMSE noise power with speech presence probability (Gerkmann & Hendriks 2012)
min_stats      minimum statistics, centred sliding minimum + bias (after Martin 2001)
mcra           minima-controlled recursive averaging (Cohen & Berdugo 2002)
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import minimum_filter1d, percentile_filter, uniform_filter1d
from scipy.signal import lfilter

from .config import Config
from .stft import frame_params

EPS = 1e-12
ESTIMATORS = ("lowest_energy", "spp", "min_stats", "mcra")


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


def _alpha(tau_sec: float, hop_sec: float) -> float:
    """Per-frame recursive smoothing factor for a time constant (0 -> no smoothing)."""
    return float(np.exp(-hop_sec / tau_sec)) if tau_sec > 0 else 0.0


def _frames(sec: float, hop_sec: float) -> int:
    return max(1, int(round(sec / hop_sec)))


def _smooth_time(x, a: float, zero_phase: bool) -> np.ndarray:
    """First-order recursive smoothing along frames; forward-backward if zero_phase."""
    if a <= 0:
        return x.copy()
    b, den = [1 - a], [1, -a]
    zi = x[:, :1] * a                                   # start at the first value
    y = lfilter(b, den, x, axis=1, zi=zi)[0]
    if zero_phase:
        y = lfilter(b, den, y[:, ::-1], axis=1, zi=y[:, -1:] * a)[0][:, ::-1]
    return y


def _spp_track(power, init, xi_h1, a_n, a_p, order):
    """One pass of the SPP noise tracker over the frames in the given order."""
    lam = init.copy()
    p_bar = np.full_like(init, 0.5)
    out = np.empty_like(power)
    ratio = 1.0 + xi_h1
    for k in order:
        gamma = power[:, k] / lam
        p1 = 1.0 / (1.0 + ratio * np.exp(-np.minimum(gamma * xi_h1 / ratio, 700.0)))
        p_bar = a_p * p_bar + (1 - a_p) * p1
        p1 = np.where(p_bar > 0.99, np.minimum(p1, 0.99), p1)   # avoid stagnation
        lam = a_n * lam + (1 - a_n) * ((1 - p1) * power[:, k] + p1 * lam)
        lam = np.maximum(lam, EPS)
        out[:, k] = lam
    return out


def spp(power, init, cfg: Config, hop_sec: float) -> np.ndarray:
    xi_h1 = 10 ** (cfg.spp_xi_h1_db / 10)
    a_n = _alpha(cfg.spp_tau_sec, hop_sec)
    a_p = 0.9 ** (hop_sec / 0.016)                      # paper: 0.9 per 16 ms hop
    n = power.shape[1]
    fwd = _spp_track(power, init, xi_h1, a_n, a_p, range(n))
    if not cfg.noise_bidirectional:
        return fwd
    bwd = _spp_track(power, init, xi_h1, a_n, a_p, range(n - 1, -1, -1))
    return np.sqrt(fwd * bwd)                           # geometric mean of both passes


def min_stats(power, init, cfg: Config, hop_sec: float) -> np.ndarray:
    smooth = _smooth_time(power, _alpha(cfg.ms_tau_sec, hop_sec), zero_phase=True)
    win = _frames(cfg.ms_window_sec, hop_sec)
    p_min = minimum_filter1d(smooth, size=win, axis=1, mode="nearest")
    # the sliding minimum is blocky; smooth it over the same span
    p_min = uniform_filter1d(p_min, size=win, axis=1, mode="nearest")
    return cfg.ms_bias * p_min


def mcra(power, init, cfg: Config, hop_sec: float) -> np.ndarray:
    # S: power smoothed over 3 bins and over time; S_min: its sliding minimum
    s = uniform_filter1d(power, size=3, axis=0, mode="nearest")
    s = _smooth_time(s, _alpha(cfg.mcra_tau_s_sec, hop_sec), zero_phase=True)
    win = _frames(cfg.ms_window_sec, hop_sec)
    s_min = minimum_filter1d(s, size=win, axis=1, mode="nearest")
    speech = (s > cfg.mcra_delta * s_min).astype(float)
    p = _smooth_time(speech, _alpha(cfg.mcra_tau_p_sec, hop_sec), zero_phase=True)
    a_d = _alpha(cfg.mcra_tau_d_sec, hop_sec)
    a = a_d + (1 - a_d) * p                              # update stops while speech is likely

    def track(order):
        lam, out = init.copy(), np.empty_like(power)
        for k in order:
            lam = a[:, k] * lam + (1 - a[:, k]) * power[:, k]
            out[:, k] = lam
        return out

    n = power.shape[1]
    est = track(range(n))
    if cfg.noise_bidirectional:
        est = np.sqrt(est * track(range(n - 1, -1, -1)))
    return cfg.mcra_bias * est


def quantile(power, cfg: Config, hop_sec: float) -> np.ndarray:
    """Quantile-based noise PSD (after Stahl, Fischer & Bippus 2000).

    Speech is sparse in each frequency bin, so a low quantile over time of |Y|^2
    in that bin is mostly noise. For noise-only frames |D|^2 is exponential with
    mean lambda, whose q-quantile is -ln(1-q)*lambda, which gives the bias
    correction. quant_window_sec = 0 uses the whole file (stationary estimate);
    otherwise a centred sliding window follows slow changes.
    """
    q = cfg.quant_q
    if cfg.quant_smooth_bins > 1:                        # less variance before the quantile
        power = uniform_filter1d(power, size=cfg.quant_smooth_bins, axis=0, mode="nearest")
    win = _frames(cfg.quant_window_sec, hop_sec) if cfg.quant_window_sec > 0 else 0
    if win == 0 or win >= power.shape[1]:
        est = np.quantile(power, q, axis=1, keepdims=True) * np.ones_like(power)
    else:
        est = percentile_filter(power, 100 * q, size=(1, win), mode="nearest")
        est = uniform_filter1d(est, size=win // 2 + 1, axis=1, mode="nearest")
    return cfg.quant_bias * est / -np.log1p(-q)


def estimate(z, t, sr, length, cfg: Config) -> np.ndarray:
    """(freq_bins, frames) noise PSD from the noisy STFT, by cfg.noise_estimator."""
    power = np.abs(z) ** 2
    name = cfg.noise_estimator
    _, hop0 = frame_params(sr, cfg)
    if name == "quantile":
        return np.maximum(cfg.noise_scale * quantile(power, cfg, hop0 / sr), EPS)
    if cfg.noise_init == "quantile":                     # seed for the recursive trackers
        base = np.maximum(quantile(power, cfg.with_overrides(quant_window_sec=0.0),
                                   hop0 / sr)[:, 0], EPS)
    else:
        base = lowest_energy_frames(z, t, sr, length, cfg)
    if name == "lowest_energy":
        return np.broadcast_to(cfg.noise_scale * base[:, None], power.shape)
    _, hop = frame_params(sr, cfg)
    hop_sec = hop / sr
    if name == "spp":
        est = spp(power, base, cfg, hop_sec)
    elif name == "min_stats":
        est = min_stats(power, base, cfg, hop_sec)
    elif name == "mcra":
        est = mcra(power, base, cfg, hop_sec)
    else:
        raise ValueError(f"Unknown noise estimator: {name}")
    if cfg.noise_smooth_bins > 1:
        est = uniform_filter1d(est, size=cfg.noise_smooth_bins, axis=0, mode="nearest")
    return np.maximum(cfg.noise_scale * est, EPS)
