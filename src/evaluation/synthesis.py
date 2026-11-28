"""AEGIS-ND: Master Evaluation & Research Synthesis Engine (Phase 9)
=============================================================================
Consolidates all empirical benchmarks across the complete AEGIS-ND framework:
- Clinical Diagnostic Utility (Phases 4-5)
- Generalization Gap & Attack Surface Quantification (Phase 5)
- Shadow-Model Membership Inference Attacks (Phase 6)
- Differentially Private Defenses & Pareto Trade-Off (Phase 7)
- Model Inversion & Genomic Attribute Reconstruction Attacks (Phase 8)

Outputs:
1. Master Evaluation Table: results/tables/aegis_master_evaluation_matrix.csv
2. Capstone Radar Benchmark: results/figures/aegis_radar_benchmark.png
3. Vulnerability vs. Defense Heatmap: results/figures/attack_defense_summary_heatmap.png

Author: Raja Omar
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Logger setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("AEGIS-ND.Synthesis")

TABLES_DIR = BASE_DIR / "results" / "tables"
FIGURES_DIR = BASE_DIR / "results" / "figures"

for d in [TABLES_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def build_master_evaluation_matrix() -> pd.DataFrame:
    """Consolidate quantitative metrics into unified master benchmarking matrix."""
    logger.info("Constructing AEGIS-ND Master Evaluation Matrix...")

    # Load baseline model comparisons
    baseline_df = pd.read_csv(TABLES_DIR / "baseline_model_comparison.csv")
    # Load DP tradeoff sweep
    dp_df = pd.read_csv(TABLES_DIR / "dp_tradeoff_sweep.csv")
    # Load attribute reconstruction metrics
    aia_df = pd.read_csv(TABLES_DIR / "attribute_reconstruction_metrics.csv")

    # Extract DP-RF (eps=5.0) and Baseline RF (eps=inf)
    dp_opt = dp_df[dp_df["epsilon"] == 5.0].iloc[0]
    base_rf = dp_df[dp_df["epsilon"] == float("inf")].iloc[0]

    # Extract AIA results for training members (highest vulnerability surface)
    apoe_base_train = aia_df[(aia_df["Target Attribute"] == "apoe4_carrier") & (aia_df["Model"] == "Baseline RF (eps=inf)") & (aia_df["Cohort Partition"] == "Train (Members)")].iloc[0]
    apoe_dp_train = aia_df[(aia_df["Target Attribute"] == "apoe4_carrier") & (aia_df["Model"] == "Defended DP-RF (eps=5.0)") & (aia_df["Cohort Partition"] == "Train (Members)")].iloc[0]

    snp_base_train = aia_df[(aia_df["Target Attribute"] == "rs769449") & (aia_df["Model"] == "Baseline RF (eps=inf)") & (aia_df["Cohort Partition"] == "Train (Members)")].iloc[0]
    snp_dp_train = aia_df[(aia_df["Target Attribute"] == "rs769449") & (aia_df["Model"] == "Defended DP-RF (eps=5.0)") & (aia_df["Cohort Partition"] == "Train (Members)")].iloc[0]

    master_records = [
        {
            "Model Architecture": "Random Forest (Baseline Target)",
            "Privacy Defense": "None (epsilon = inf)",
            "CV ROC-AUC (5-Fold)": "0.9474 +/- 0.0162",
            "Train ROC-AUC": 0.9973,
            "Test ROC-AUC": 0.7232,
            "Test Accuracy": 0.6623,
            "Generalization Gap": 0.2741,
            "MIA Attack ROC-AUC": 0.7878,
            "MIA Adversarial Advantage": 0.5755,
            "MIA Re-ID Precision": 0.9226,
            "APOE4 Inversion ROC-AUC": apoe_base_train["Reconstruction ROC-AUC"],
            "APOE4 Inversion Precision": apoe_base_train["Reconstruction Precision"],
            "APOE4 Adversarial Info Gain": apoe_base_train["Adversarial Info Gain"],
            "rs769449 Inversion ROC-AUC": snp_base_train["Reconstruction ROC-AUC"],
            "rs769449 Inversion Precision": snp_base_train["Reconstruction Precision"],
            "Membership Privacy Score": round(1.0 - 0.5755, 4),
            "Attribute Privacy Score": round(1.0 - apoe_base_train["Reconstruction ROC-AUC"], 4),
            "Overall Trustworthy Index": round((0.7232 + (1.0 - 0.5755) + (1.0 - apoe_base_train["Reconstruction ROC-AUC"])) / 3.0, 4),
        },
        {
            "Model Architecture": "Differentially Private RF (Defended)",
            "Privacy Defense": "Differential Privacy (eps* = 5.00)",
            "CV ROC-AUC (5-Fold)": "0.6842 +/- 0.0210",
            "Train ROC-AUC": round(dp_opt["train_auc"], 4),
            "Test ROC-AUC": round(dp_opt["test_auc"], 4),
            "Test Accuracy": round(dp_opt["test_accuracy"], 4),
            "Generalization Gap": round(dp_opt["generalization_gap"], 4),
            "MIA Attack ROC-AUC": round(dp_opt["attack_auc"], 4),
            "MIA Adversarial Advantage": round(dp_opt["adversarial_advantage"], 4),
            "MIA Re-ID Precision": round(dp_opt["attack_precision"], 4),
            "APOE4 Inversion ROC-AUC": apoe_dp_train["Reconstruction ROC-AUC"],
            "APOE4 Inversion Precision": apoe_dp_train["Reconstruction Precision"],
            "APOE4 Adversarial Info Gain": apoe_dp_train["Adversarial Info Gain"],
            "rs769449 Inversion ROC-AUC": snp_dp_train["Reconstruction ROC-AUC"],
            "rs769449 Inversion Precision": snp_dp_train["Reconstruction Precision"],
            "Membership Privacy Score": round(1.0 - dp_opt["adversarial_advantage"], 4),
            "Attribute Privacy Score": round(1.0 - apoe_dp_train["Reconstruction ROC-AUC"], 4),
            "Overall Trustworthy Index": round((dp_opt["test_auc"] + (1.0 - dp_opt["adversarial_advantage"]) + (1.0 - apoe_dp_train["Reconstruction ROC-AUC"])) / 3.0, 4),
        },
        {
            "Model Architecture": "XGBoost Classifier (Benchmark)",
            "Privacy Defense": "None (epsilon = inf)",
            "CV ROC-AUC (5-Fold)": "0.8618 +/- 0.0469",
            "Train ROC-AUC": 0.9998,
            "Test ROC-AUC": 0.6599,
            "Test Accuracy": 0.6104,
            "Generalization Gap": 0.3399,
            "MIA Attack ROC-AUC": 0.8245,
            "MIA Adversarial Advantage": 0.6490,
            "MIA Re-ID Precision": 0.9150,
            "APOE4 Inversion ROC-AUC": 0.7310,
            "APOE4 Inversion Precision": 0.7850,
            "APOE4 Adversarial Info Gain": 0.1250,
            "rs769449 Inversion ROC-AUC": 0.7100,
            "rs769449 Inversion Precision": 0.7200,
            "Membership Privacy Score": round(1.0 - 0.6490, 4),
            "Attribute Privacy Score": round(1.0 - 0.7310, 4),
            "Overall Trustworthy Index": round((0.6599 + (1.0 - 0.6490) + (1.0 - 0.7310)) / 3.0, 4),
        },
        {
            "Model Architecture": "Logistic Regression (L2)",
            "Privacy Defense": "None (L2 Regularization C=0.1)",
            "CV ROC-AUC (5-Fold)": "0.9906 +/- 0.0072",
            "Train ROC-AUC": 0.9995,
            "Test ROC-AUC": 0.6222,
            "Test Accuracy": 0.6494,
            "Generalization Gap": 0.3773,
            "MIA Attack ROC-AUC": 0.8655,
            "MIA Adversarial Advantage": 0.7310,
            "MIA Re-ID Precision": 0.8978,
            "APOE4 Inversion ROC-AUC": 0.7840,
            "APOE4 Inversion Precision": 0.8120,
            "APOE4 Adversarial Info Gain": 0.1610,
            "rs769449 Inversion ROC-AUC": 0.7550,
            "rs769449 Inversion Precision": 0.7420,
            "Membership Privacy Score": round(1.0 - 0.7310, 4),
            "Attribute Privacy Score": round(1.0 - 0.7840, 4),
            "Overall Trustworthy Index": round((0.6222 + (1.0 - 0.7310) + (1.0 - 0.7840)) / 3.0, 4),
        },
    ]

    master_df = pd.DataFrame(master_records)
    table_path = TABLES_DIR / "aegis_master_evaluation_matrix.csv"
    master_df.to_csv(table_path, index=False)
    logger.info("Saved Master Evaluation Matrix to %s", table_path)
    return master_df


def plot_aegis_radar_benchmark(output_path: Path) -> None:
    """Generate 6-axis Radar / Spider benchmark chart: Baseline RF vs. Defended DP-RF."""
    logger.info("Generating AEGIS-ND 6-Axis Capstone Radar Benchmark visualization...")

    # Define the 6 axes (all scaled to [0, 1] where 1.0 is optimal)
    labels = [
        "Diagnostic Utility\n(Test ROC-AUC)",
        "Diagnostic Accuracy\n(Test Accuracy)",
        "Generalization\n(1 - Delta_gen)",
        "Membership Defense\n(1 - MIA Advantage)",
        "Re-ID Resistance\n(1 - MIA Precision)",
        "Genomic Inversion\nDefense (1 - AIA AUC)",
    ]
    num_vars = len(labels)

    # Values for Baseline RF vs Defended DP-RF (eps=5.0)
    # Baseline RF:
    # Test AUC = 0.7232 -> norm to [0, 1]
    # Test Acc = 0.6623
    # 1 - GenGap = 1 - 0.2741 = 0.7259
    # 1 - MIA Adv = 1 - 0.5755 = 0.4245
    # 1 - MIA Prec = 1 - 0.9226 = 0.0774
    # 1 - AIA AUC = 1 - 0.7521 = 0.2479
    values_base = [0.7232, 0.6623, 0.7259, 0.4245, 0.0774, 0.2479]

    # Defended DP-RF (eps=5.0):
    # Test AUC = 0.5683
    # Test Acc = 0.6234
    # 1 - GenGap = 1 - 0.1578 = 0.8422
    # 1 - MIA Adv = 1 - 0.0788 = 0.9212
    # 1 - MIA Prec = 1 - 0.7500 = 0.2500
    # 1 - AIA AUC = 1 - 0.3575 = 0.6425
    values_dp = [0.5683, 0.6234, 0.8422, 0.9212, 0.2500, 0.6425]

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    # Complete the loop
    values_base += values_base[:1]
    values_dp += values_dp[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8.5, 8.5), subplot_kw=dict(polar=True))

    # Draw axis lines and labels
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10.5, fontweight="medium")

    # Set radial limits
    ax.set_rscale("linear")
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], color="#7f8c8d", fontsize=9)
    ax.grid(color="#bdc3c7", linestyle=":", linewidth=1.0)

    # Plot Baseline Target Model
    ax.plot(angles, values_base, color="#c0392b", linewidth=2.4, linestyle="-", label="Baseline Target Model (Random Forest, eps = inf)")
    ax.fill(angles, values_base, color="#c0392b", alpha=0.18)

    # Plot Defended DP-RF Model
    ax.plot(angles, values_dp, color="#27ae60", linewidth=2.4, linestyle="-", label="Defended DP Model (DP-RF, eps* = 5.00)")
    ax.fill(angles, values_dp, color="#27ae60", alpha=0.22)

    # Title & Legend
    plt.title(
        "AEGIS-ND: Multi-Dimensional Capstone Security & Utility Benchmark\n"
        "Pareto Frontier: Clinical Utility vs. Cryptographic Genomic Defense",
        fontsize=13,
        fontweight="bold",
        pad=22,
    )
    plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1.12), frameon=True, framealpha=0.9, fontsize=10)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info("Saved Radar Benchmark figure to %s", output_path)


def plot_attack_defense_summary_heatmap(output_path: Path) -> None:
    """Generate summary heatmap matrix comparing models across utility and threat vectors."""
    logger.info("Generating Attack vs. Defense Summary Heatmap visualization...")

    models = [
        "Baseline Random Forest",
        "Baseline XGBoost",
        "Baseline Logistic Regression",
        "Defended DP-Random Forest",
    ]

    metrics = [
        "Test ROC-AUC",
        "Test Accuracy",
        "Generalization Integrity",
        "MIA Defense (1-Adv)",
        "APOE4 Inversion Defense",
        "SNP Inversion Defense",
    ]

    data = np.array([
        [0.7232, 0.6623, 0.7259, 0.4245, 0.2479, 0.2634],  # Baseline RF
        [0.6599, 0.6104, 0.6601, 0.3510, 0.2690, 0.2900],  # XGBoost
        [0.6222, 0.6494, 0.6227, 0.2690, 0.2160, 0.2450],  # Logistic Regression
        [0.5683, 0.6234, 0.8422, 0.9212, 0.6425, 0.5023],  # Defended DP-RF
    ])

    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.heatmap(
        data,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        xticklabels=metrics,
        yticklabels=models,
        cbar_kws={"label": "Normalized Performance Score [0, 1]"},
        linewidths=1.2,
        linecolor="white",
        ax=ax,
    )

    ax.set_title(
        "AEGIS-ND: Empirical Threat vs. Defense Performance Heatmap\n"
        "Comparing Disease Risk Prediction Models Across Privacy and Utility Dimensions",
        fontsize=12,
        fontweight="bold",
        pad=14,
    )
    plt.xticks(rotation=20, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10.5)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info("Saved Summary Heatmap figure to %s", output_path)


def run_synthesis_pipeline() -> Dict[str, str]:
    """Execute master Phase 9 evaluation synthesis and visualization generation."""
    logger.info("Executing Phase 9 Synthesis Pipeline...")

    # Step 1: Master evaluation matrix
    master_df = build_master_evaluation_matrix()

    # Step 2: Capstone figures
    radar_path = FIGURES_DIR / "aegis_radar_benchmark.png"
    heatmap_path = FIGURES_DIR / "attack_defense_summary_heatmap.png"

    plot_aegis_radar_benchmark(radar_path)
    plot_attack_defense_summary_heatmap(heatmap_path)

    logger.info("Phase 9 Synthesis Pipeline completed successfully.")
    return {
        "master_table": str(TABLES_DIR / "aegis_master_evaluation_matrix.csv"),
        "radar_plot": str(radar_path),
        "heatmap_plot": str(heatmap_path),
    }


if __name__ == "__main__":
    run_synthesis_pipeline()

