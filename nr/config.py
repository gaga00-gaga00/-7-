"""Algorithm parameters and JSON config loading."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

from scipy import signal


@dataclass(frozen=True)
class Config:
    # STFT, in seconds so other sample rates work too (512 / 128 samples @ 16 kHz)
    win_sec: float = 0.032
    hop_sec: float = 0.008
    # basic filter (stages of the combined filter; each is also its own method)
    filter_order: int = 4
    lowpass_hz: float = 3500.0
    highpass_hz: float = 100.0
    bandstop_hz: tuple = (950.0, 1050.0)
    basic_stages: tuple = ("highpass", "lowpass", "bandstop")
    # noise estimate (course baseline: mean of the lowest-energy frames)
    noise_fraction: float = 0.20
    # spectral subtraction
    subtraction_alpha: float = 1.5
    subtraction_floor: float = 0.02
    # Wiener / decision-directed a-priori SNR
    dd_alpha: float = 0.98
    wiener_gain_floor: float = 0.05
    # MMSE-LSA
    lsa_xi_min_db: float = -25.0
    lsa_gain_floor: float = 0.05
    # time-varying noise PSD (nr/noise.py), used by omlsa:
    # "lowest_energy" | "spp" | "min_stats" | "mcra"
    noise_estimator: str = "spp"
    noise_bidirectional: bool = False  # run the recursive trackers both ways in time
    noise_smooth_bins: int = 1         # frequency smoothing of the tracked PSD (1 = off)
    noise_scale: float = 1.0           # over-/under-estimation factor on the noise PSD
    noise_init: str = "lowest_energy"  # seed of the recursive trackers: "lowest_energy" | "quantile"
    # quantile noise estimate (Stahl et al. 2000): per-bin low quantile over time
    quant_q: float = 0.25
    quant_window_sec: float = 0.0      # 0 = whole file (stationary), else sliding window
    quant_smooth_bins: int = 1         # frequency smoothing of |Y|^2 before the quantile
    quant_bias: float = 1.0
    spp_xi_h1_db: float = 15.0    # fixed a-priori SNR under speech presence
    spp_tau_sec: float = 0.3      # noise PSD smoothing time constant (paper: 0.8 per 16 ms = 0.07 s)
    ms_tau_sec: float = 0.1       # minimum statistics: power smoothing
    ms_window_sec: float = 1.5    # minimum search window (also MCRA)
    ms_bias: float = 1.5          # minimum -> mean bias compensation
    mcra_tau_s_sec: float = 0.07  # MCRA: power smoothing
    mcra_tau_p_sec: float = 0.01  # speech presence smoothing
    mcra_tau_d_sec: float = 0.3   # noise PSD smoothing
    mcra_delta: float = 5.0       # S / S_min threshold for speech presence
    mcra_bias: float = 1.0
    # OM-LSA (Cohen & Berdugo 2001): G = G_lsa^p * G_min^(1-p)
    omlsa_gain_floor: float = 0.1
    omlsa_q: float = 0.5          # a-priori speech absence probability
    omlsa_smooth_bins: int = 3    # frequency smoothing of the SNR used for p
    omlsa_tsnr: bool = False      # two-step a-priori SNR refinement (Plapous et al. 2006)
    dd_combine: str = "forward"   # "forward" | "geomean" | "max": offline two-direction DD
    hrnr_weight: float = 0.0      # harmonic regeneration second gain (0 = off, 1 = full)
    # category-specific additions on top of the tracked noise PSD (nr/noise_modules.py);
    # weight 0 = off, so the defaults leave omlsa unchanged
    tone_weight: float = 0.0      # C1: lift the noise PSD along noise-tone tracks
    impulse_weight: float = 0.0   # C2: lift the noise PSD inside short broadband bursts
    post_highpass_hz: float = 0.0  # C3: high-pass after enhancement (0 = off)
    # input-adaptive blend (nr/blend.py, method "adaptive"); thresholds chosen on train
    ad_fixed_flatness: float = 0.4     # F1 above: blend towards the per-file fixed noise PSD
    ad_fast_flatness: float = 0.25     # F1 below: faster SPP tracking
    ad_flatness_width: float = 0.03    # sigmoid width for both F1 memberships
    ad_fast_tau_sec: float = 0.1
    ad_strong_low_ratio: float = 0.17  # F6 below: stronger suppression
    ad_strong_noise_scale: float = 2.0
    ad_strong_xi_min_db: float = -35.0
    ad_lowcut_low_ratio: tuple = (0.4, 0.8)  # F6 inside: high-pass the output
    ad_lowcut_hz: float = 150.0
    ad_low_ratio_width: float = 0.03   # sigmoid width for the F6 memberships
    ad_min_weight: float = 0.3         # all memberships below: use the base setting

    def __post_init__(self):
        """Reject inconsistent parameters early (checks from the course notebook)."""
        n_fft = int(round(self.win_sec * 16000))
        hop = int(round(self.hop_sec * 16000))
        if not 0 < self.hop_sec <= self.win_sec or n_fft < 4:
            raise ValueError("Require 0 < hop_sec <= win_sec and a window of >= 4 samples.")
        if not signal.check_NOLA("hann", n_fft + n_fft % 2, n_fft + n_fft % 2 - hop):
            raise ValueError("STFT window/hop does not satisfy NOLA (no perfect reconstruction).")
        if not 0 < self.noise_fraction <= 1:
            raise ValueError("noise_fraction must be in (0, 1].")
        if not 0 <= self.dd_alpha < 1:
            raise ValueError("dd_alpha must be in [0, 1).")
        if not 0 <= self.subtraction_floor <= 1 or self.subtraction_alpha < 0:
            raise ValueError("Invalid spectral subtraction parameters.")
        if self.noise_estimator not in ("lowest_energy", "spp", "min_stats", "mcra", "quantile"):
            raise ValueError("Unknown noise_estimator.")
        if self.noise_scale <= 0:
            raise ValueError("noise_scale must be positive.")
        if self.noise_init not in ("lowest_energy", "quantile"):
            raise ValueError("Unknown noise_init.")
        if not 0 < self.quant_q < 1 or self.quant_window_sec < 0 or self.quant_bias <= 0 \
                or self.quant_smooth_bins < 1:
            raise ValueError("Invalid quantile noise estimator parameters.")
        for name in ("spp_tau_sec", "ms_tau_sec", "mcra_tau_s_sec", "mcra_tau_p_sec",
                     "mcra_tau_d_sec"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0.")
        if self.ms_window_sec <= 0 or self.ms_bias <= 0 or self.mcra_bias <= 0 \
                or self.mcra_delta <= 1:
            raise ValueError("Invalid minimum statistics / MCRA parameters.")
        if not 0 < self.omlsa_q < 1:
            raise ValueError("omlsa_q must be in (0, 1).")
        if self.tone_weight < 0 or self.impulse_weight < 0 or self.post_highpass_hz < 0:
            raise ValueError("tone_weight, impulse_weight and post_highpass_hz must be >= 0.")
        if self.ad_flatness_width <= 0 or self.ad_low_ratio_width <= 0                 or self.ad_fast_tau_sec < 0 or self.ad_strong_noise_scale <= 0                 or len(self.ad_lowcut_low_ratio) != 2 or self.ad_lowcut_hz <= 0:
            raise ValueError("Invalid adaptive blend parameters.")
        if self.dd_combine not in ("forward", "geomean", "max") or not 0 <= self.hrnr_weight <= 1:
            raise ValueError("Invalid dd_combine / hrnr_weight.")
        if self.noise_smooth_bins < 1 or int(self.noise_smooth_bins) != self.noise_smooth_bins:
            raise ValueError("noise_smooth_bins must be a positive integer.")
        if self.omlsa_smooth_bins < 1 or int(self.omlsa_smooth_bins) != self.omlsa_smooth_bins:
            raise ValueError("omlsa_smooth_bins must be a positive integer.")
        for name in ("wiener_gain_floor", "lsa_gain_floor", "omlsa_gain_floor"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1].")
        if not 0 < self.highpass_hz < self.lowpass_hz:
            raise ValueError("Require 0 < highpass_hz < lowpass_hz.")
        if len(self.bandstop_hz) != 2 or not 0 < self.bandstop_hz[0] < self.bandstop_hz[1]:
            raise ValueError("Invalid band-stop interval.")
        if self.filter_order < 1 or int(self.filter_order) != self.filter_order:
            raise ValueError("filter_order must be a positive integer.")
        if any(s not in ("lowpass", "highpass", "bandstop") for s in self.basic_stages):
            raise ValueError("Unknown basic filter stage.")

    def with_overrides(self, **overrides) -> "Config":
        known = {f.name for f in fields(self)}
        unknown = set(overrides) - known
        if unknown:
            raise KeyError(f"Unknown Config field(s): {sorted(unknown)}")
        fixed = {k: tuple(v) if isinstance(v, list) else v for k, v in overrides.items()}
        return replace(self, **fixed)

    def to_dict(self) -> dict:
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}


DEFAULT = Config()


def load_config(path: str | Path) -> tuple[str, Config]:
    """Read {"method": ..., "params": {...}} and return (method, Config)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["method"], DEFAULT.with_overrides(**data.get("params", {}))
