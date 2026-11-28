"""AEGIS-ND: Evaluation Visualization Suite
=============================================================================
Provides publication-grade visualization routines for the AEGIS-ND benchmark,
including multi-dimensional radar plots and attack-defense heatmaps.

Author: Raja Omar
"""

from __future__ import annotations

from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_radar_benchmark(
    labels: List[str],
    baseline_values: List[float],
    defended_values: List[float],
    output_path: Path,
    title: str = "AEGIS-ND: Multi-Dimensional Capstone Security & Utility Benchmark",
) -> None:
    """Generate a multi-dimensional 6-axis Radar / Spider benchmark chart."""
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    base_vals = baseline_values + baseline_values[:1]
    dp_vals = defended_values + defended_values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8.5, 8.5), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10.5, fontweight="medium")

    ax.set_rscale("linear")
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], color="#7f8c8d", fontsize=9)
    ax.grid(color="#bdc3c7", linestyle=":", linewidth=1.0)

    # Baseline Model
    ax.plot(angles, base_vals, color="#c0392b", linewidth=2.4, linestyle="-", label="Baseline Target Model (Random Forest, eps = inf)")
    ax.fill(angles, base_vals, color="#c0392b", alpha=0.18)

    # Defended DP Model
    ax.plot(angles, dp_vals, color="#27ae60", linewidth=2.4, linestyle="-", label="Defended DP Model (DP-RF, eps* = 5.00)")
    ax.fill(angles, dp_vals, color="#27ae60", alpha=0.22)

    plt.title(title, fontsize=13, fontweight="bold", pad=22)
    plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1.12), frameon=True, framealpha=0.9, fontsize=10)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

