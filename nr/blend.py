"""Input-adaptive OM-LSA: rule-based memberships blend the settings per file.

Two features of the noisy input (nr/features.py) drive four memberships in
[0, 1], each a sigmoid w = 1 / (1 + exp(-(F - threshold) / width)):

    fixed    flat noise spectrum (F1 high): stationary broadband noise, where the
             tracker only pulls speech into the noise estimate -> per-file fixed PSD
    fast     coloured noise (F1 low): follow the noise faster (short SPP time constant)
    strong   little low-frequency noise (F6 low): over-estimate the noise and
             suppress deeper
    lowcut   mid-range low-frequency share (F6 between two thresholds): high-pass
    hrnr_off noise almost entirely below 300 Hz (F6 very high): fade out the
             harmonic regeneration gain (cfg.hrnr_weight)

Blending (all memberships 0 = the plain omlsa method):
    noise PSD   (1 - fixed) * tracked + fixed * per-file fixed
    parameters  p = p_base + w * (p_category - p_base) for the SPP time constant,
                the noise scale and the a-priori SNR floor
    output      (1 - lowcut) * y + lowcut * highpass(y)
If every membership is below cfg.ad_min_weight the base setting is used as is.

Thresholds were chosen on train (results/stage2_train_all); no model is fitted.
"""
from __future__ import annotations

import numpy as np

from . import features as ft
from . import noise as noise_est
from .config import Config
from .stft import stft


def _sigmoid(value, threshold, width):
    return float(1.0 / (1.0 + np.exp(-np.clip((value - threshold) / width, -50, 50))))


def memberships(x, sr, cfg: Config) -> dict:
    flat, low = ft.spectrum_features(x, sr)
    w = {
        "fixed": _sigmoid(flat, cfg.ad_fixed_flatness, cfg.ad_flatness_width),
        "fast": _sigmoid(-flat, -cfg.ad_fast_flatness, cfg.ad_flatness_width),
        "strong": _sigmoid(-low, -cfg.ad_strong_low_ratio, cfg.ad_low_ratio_width),
        "lowcut": _sigmoid(low, cfg.ad_lowcut_low_ratio[0], cfg.ad_low_ratio_width)
                  * _sigmoid(-low, -cfg.ad_lowcut_low_ratio[1], cfg.ad_low_ratio_width),
    }
    if max(w.values()) < cfg.ad_min_weight:
        w = dict.fromkeys(w, 0.0)
    # not a setting to blend in but a brake on the second gain, so it is kept
    # out of the safeguard above
    w["hrnr_off"] = _sigmoid(low, cfg.ad_hrnr_off_low_ratio, cfg.ad_low_ratio_width)
    return w


def adaptive(x, sr, cfg: Config):
    from .methods import butterworth_stages, synthesize  # methods imports this module

    w = memberships(x, sr, cfg)
    strong = w["strong"] * (1.0 - w["fixed"])      # a flat spectrum takes precedence
    tau = cfg.spp_tau_sec + w["fast"] * (cfg.ad_fast_tau_sec - cfg.spp_tau_sec)
    xi_min = cfg.lsa_xi_min_db + strong * (cfg.ad_strong_xi_min_db - cfg.lsa_xi_min_db)

    t, z = stft(x, sr, cfg)
    noise = noise_est.estimate(z, t, sr, len(x), cfg.with_overrides(spp_tau_sec=tau))
    if w["fixed"] > 0:
        fixed = cfg.noise_scale * noise_est.lowest_energy_frames(z, t, sr, len(x), cfg)[:, None]
        noise = (1.0 - w["fixed"]) * noise + w["fixed"] * fixed
    noise = noise * (1.0 + strong * (cfg.ad_strong_noise_scale - 1.0))

    hrnr = cfg.hrnr_weight
    if cfg.ad_hrnr_flat_weight >= 0:               # flat noise takes more regeneration
        hrnr += w["fixed"] * (cfg.ad_hrnr_flat_weight - hrnr)
    y = synthesize(z, noise, sr, len(x), cfg.with_overrides(
        lsa_xi_min_db=xi_min, hrnr_weight=hrnr * (1.0 - w["hrnr_off"])))
    if w["lowcut"] > 0:
        cut = butterworth_stages(y, sr, cfg.with_overrides(highpass_hz=cfg.ad_lowcut_hz),
                                 ("highpass",))
        y = (1.0 - w["lowcut"]) * y + w["lowcut"] * cut
    return y
