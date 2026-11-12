"""Dataset overview (course notebook dataset_exploration.ipynb, sections 2-3).

    python tools/explore_dataset.py

Prints and saves to results/figures/dataset/ (git-ignored):
  summary.csv          per split: pairs, WAV files, speakers, sample rates,
                       duration min/mean/max, total minutes, noise types, SNRs
  noise_counts.csv     samples per noise category and split
  noise_details.csv    per split and category: samples, noise codes, SNR choices
  difficulty_snr.csv   samples per difficulty x target SNR and split
  noise_counts.png     bar chart of noise_counts
To look at single pairs (clean / noisy / added noise), use
  python tools/spectrogram.py <segment_id> --added-noise --no-enhance
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate import load_manifest  # noqa: E402

SPLITS = ("train", "validation")


def summarize_dataset(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    groups = list(datasets.items()) + [("all", pd.concat(datasets.values(), ignore_index=True))]
    for split, table in groups:
        duration = table["num_samples"] / table["sample_rate_hz"]
        speakers = table["segment_id"].str.extract(r"^(speaker\d+)_", expand=False)
        rows.append({
            "split": split,
            "pairs": len(table),
            "wav_files": 2 * len(table),
            "speakers": speakers.nunique(),
            "sample_rates_hz": sorted(table["sample_rate_hz"].unique().tolist()),
            "duration_min_mean_max_s": [round(v, 3) for v in
                                        (duration.min(), duration.mean(), duration.max())],
            "total_minutes": round(duration.sum() / 60, 2),
            "noise_categories": table["noise_category"].nunique(),
            "target_snr_db": sorted(table["target_snr_db"].unique().tolist()),
        })
    return pd.DataFrame(rows).set_index("split")


def main():
    datasets = {s: load_manifest(ROOT, s, None) for s in SPLITS}
    out = ROOT / "results" / "figures" / "dataset"
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 80)

    summary = summarize_dataset(datasets)
    summary.to_csv(out / "summary.csv", encoding="utf-8-sig")
    print(summary.to_string(), "\n")

    train_speakers = set(datasets["train"].segment_id.str.extract(r"^(speaker\d+)_")[0])
    val_speakers = set(datasets["validation"].segment_id.str.extract(r"^(speaker\d+)_")[0])
    print(f"speakers in both splits: {len(train_speakers & val_speakers)} "
          f"(0 = validation speakers are unseen, like the hidden test set)\n")

    all_samples = pd.concat(datasets.values(), ignore_index=True)
    counts = pd.crosstab(all_samples.noise_category, all_samples["split"]) \
        .reindex(columns=list(SPLITS), fill_value=0).sort_index()
    counts.to_csv(out / "noise_counts.csv", encoding="utf-8-sig")
    print(counts.to_string(), "\n")

    details = all_samples.groupby(["split", "noise_category"]).agg(
        samples=("segment_id", "size"),
        noise_codes=("noise_code", lambda x: ", ".join(sorted(x.unique()))),
        difficulty=("noise_difficulty", lambda x: ", ".join(sorted(x.unique()))),
        snr_choices_db=("target_snr_db", lambda x: sorted(x.unique().tolist())),
    ).reset_index()
    details.to_csv(out / "noise_details.csv", index=False, encoding="utf-8-sig")

    grid = pd.crosstab([all_samples["split"], all_samples.noise_difficulty],
                       all_samples.target_snr_db)
    grid.to_csv(out / "difficulty_snr.csv", encoding="utf-8-sig")
    print(grid.to_string(), "\n")

    train_noise = set(datasets["train"].noise_code)
    unseen = sorted(set(datasets["validation"].noise_code) - train_noise)
    print(f"validation noise codes not in train: {len(unseen)} {unseen}\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), sharey=True, constrained_layout=True)
    for ax, split, color in zip(axes, SPLITS, ("#426FA6", "#159DAD")):
        ax.barh(counts.index, counts[split], color=color)
        ax.set(title=split.capitalize(), xlabel="Number of samples")
        ax.grid(axis="x", alpha=0.2)
    axes[0].invert_yaxis()
    fig.savefig(out / "noise_counts.png", dpi=110)
    plt.close(fig)
    print("saved:", out)


if __name__ == "__main__":
    main()
