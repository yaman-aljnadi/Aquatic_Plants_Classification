"""One figure for the professor meeting: pooled accuracy, plain labels.

    python scripts/plot_protonet_meeting.py

Numbers are the five-seed means already reported in Reports/episodic_protonet_results.md.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "..", "Reports", "protonet_meeting.png")

# Pooled test, all training photos as support, mean +/- std over seeds 8-12.
BARS = [
    ("No training\n(averages only)", 42.3, 2.3, "#b0b7c3"),
    ("Trained", 50.8, 4.0, "#7aa2d6"),
    ("Trained,\nlast block learns", 57.5, 4.3, "#3d7ab5"),
    ("Trained\n+ leaf detail", 59.3, 4.8, "#1f4e79"),
    ("Older\nlinear model", 59.5, 4.6, "#8d8d8d"),
    ("Best older\nmodel", 65.1, 5.0, "#4a4a4a"),
]

# Best new model (trained + leaf detail, last block learns), by support size.
SHOTS = [("1 photo", 58.0, 3.7), ("5 photos", 60.4, 5.1), ("10 photos", 59.3, 4.5), ("All photos", 59.3, 4.8)]


def main():
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), gridspec_kw={"width_ratios": [1.35, 1]})

    ax = axes[0]
    xs = range(len(BARS))
    means = [row[1] for row in BARS]
    stds = [row[2] for row in BARS]
    colors = [row[3] for row in BARS]
    ax.bar(xs, means, yerr=stds, color=colors, capsize=3, width=0.72, error_kw={"elinewidth": 1})
    for x, mean in zip(xs, means):
        ax.text(x, mean + 6.2, f"{mean:.0f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([row[0] for row in BARS], fontsize=8)
    ax.set_ylim(0, 82)
    ax.set_ylabel("Photos labeled correctly (%)")
    ax.set_title("Mixed test, using every training photo")
    ax.axhline(59.5, color="#8d8d8d", lw=0.8, ls="--")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    ax = axes[1]
    xs = range(len(SHOTS))
    means = [row[1] for row in SHOTS]
    stds = [row[2] for row in SHOTS]
    ax.bar(xs, means, yerr=stds, color="#1f4e79", capsize=3, width=0.65, error_kw={"elinewidth": 1})
    for x, mean in zip(xs, means):
        ax.text(x, mean + 4.2, f"{mean:.0f}%", ha="center", va="bottom", fontsize=9)
    ax.axhline(65.1, color="#4a4a4a", lw=1, ls="--", label="Best older model (65%)")
    ax.axhline(59.5, color="#8d8d8d", lw=1, ls=":", label="Older linear model (60%)")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([row[0] for row in SHOTS], fontsize=9)
    ax.set_ylim(0, 82)
    ax.set_ylabel("Photos labeled correctly (%)")
    ax.set_title("Best new model, fewer example photos")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
    fig.savefig(OUT, dpi=160)
    print(OUT)


if __name__ == "__main__":
    main()
