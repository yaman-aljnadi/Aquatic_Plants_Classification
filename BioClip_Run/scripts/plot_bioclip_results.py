"""Meeting figure for BioCLIP vs ConvNeXt.

    From BioClip_Run/:
      ..\\Yaman-Aquatic_Plants_Classification\\.venv\\Scripts\\python.exe scripts\\plot_bioclip_results.py

Numbers are the five-seed means from the official Week 13 runs (hardcoded, same
style as the ProtoNet meeting plot).
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PNG = os.path.join(ROOT, "figs", "bioclip_meeting.png")
OUT_PDF = os.path.join(ROOT, "figs", "bioclip_meeting.pdf")

# Mixed / pooled test accuracy, mean +/- std over seeds 8-12.
POOLED = [
    ("ConvNeXt\nfrozen", 59.5, 4.6, "#8d8d8d"),
    ("BioCLIP\nfrozen", 64.9, 4.0, "#7aa2d6"),
    ("BioCLIP\nlast block", 72.5, 4.5, "#1f4e79"),
    ("ConvNeXt\nlast stage", 65.1, 5.0, "#4a4a4a"),
]

# Lab vs web for the two "best" setups.
GROUPED = {
    "labels": ["Lab photos", "Web photos"],
    "BioCLIP last block": {"means": [89.2, 67.1], "stds": [4.9, 5.4], "color": "#1f4e79"},
    "ConvNeXt last stage": {"means": [85.6, 58.1], "stds": [9.7, 4.8], "color": "#8d8d8d"},
}


def main():
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})

    ax = axes[0]
    xs = np.arange(len(POOLED))
    means = [row[1] for row in POOLED]
    stds = [row[2] for row in POOLED]
    colors = [row[3] for row in POOLED]
    ax.bar(xs, means, yerr=stds, color=colors, capsize=3, width=0.72, error_kw={"elinewidth": 1})
    for x, mean in zip(xs, means):
        ax.text(x, mean + 6.5, f"{mean:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.axhline(65.1, color="#4a4a4a", lw=0.9, ls="--", label="Old best mixed (65.1%)")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([row[0] for row in POOLED], fontsize=8)
    ax.set_ylim(0, 95)
    ax.set_ylabel("Photos labeled correctly (%)")
    ax.set_title("Mixed test: which encoder wins?")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    ax = axes[1]
    labels = GROUPED["labels"]
    x = np.arange(len(labels))
    width = 0.35
    series = [
        ("BioCLIP last block", GROUPED["BioCLIP last block"]),
        ("ConvNeXt last stage", GROUPED["ConvNeXt last stage"]),
    ]
    for i, (name, data) in enumerate(series):
        offset = (i - 0.5) * width
        bars = ax.bar(
            x + offset,
            data["means"],
            width,
            yerr=data["stds"],
            color=data["color"],
            capsize=3,
            label=name,
            error_kw={"elinewidth": 1},
        )
        for rect, mean in zip(bars, data["means"]):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                mean + 5.5,
                f"{mean:.0f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Photos labeled correctly (%)")
    ax.set_title("Lab vs web (best BioCLIP vs best ConvNeXt)")
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=160)
    fig.savefig(OUT_PDF)
    print(OUT_PNG)
    print(OUT_PDF)


if __name__ == "__main__":
    main()
