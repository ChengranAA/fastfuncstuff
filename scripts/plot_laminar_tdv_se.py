#!/usr/bin/env python3
"""Plot depth-wise GE/SE validation from bench_laminar_tdv_se report.json."""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("report")
    p.add_argument("output")
    args = p.parse_args()
    with open(args.report) as f:
        rows = json.load(f)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True, layout="constrained")
    for ax, subject in zip(axes.flat, sorted({r["subject"] for r in rows}), strict=True):
        group = [
            r
            for r in rows
            if r["subject"] == subject
            and r["selection"] == "ge_positive"
            and r["hi"] - r["lo"] < 0.3
        ]
        local = sorted([r for r in group if r["target"] == "local"], key=lambda r: r["lo"])
        micro = sorted([r for r in group if r["target"] == "micro"], key=lambda r: r["lo"])
        x = np.array([(r["lo"] + r["hi"]) / 2 for r in local])
        for name, key, which, color, ls in [
            ("Original GE", "temporal_raw", local, "#444444", "-"),
            ("Transport-corrected", "temporal_corrected", local, "#276FBF", "-"),
            ("Microvascular target", "temporal_corrected", micro, "#B8336A", "-"),
            ("Smoothing control", "temporal_smooth", local, "#32936F", "--"),
        ]:
            ax.plot(
                x,
                [r[key] for r in which],
                label=name,
                color=color,
                linestyle=ls,
                marker="o",
                markersize=4,
            )
        ax.set_title(f"sub-{subject}")
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1], ["pial", ".2", ".4", ".6", ".8", "white"])
        ax.grid(alpha=0.2)
    for ax in axes[:, 0]:
        ax.set_ylabel("Median voxelwise correlation with SE")
    for ax in axes[-1]:
        ax.set_xlabel("Segmentation-derived cortical depth")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Continuous vascular deveining: external spin-echo check", fontsize=14)
    fig.savefig(args.output, dpi=180)


if __name__ == "__main__":
    main()
