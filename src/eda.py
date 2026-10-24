"""AEGIS-ND: Exploratory Data Analysis & Biological Validation (Phase 4)
=====================================================================
Generates publication-grade figures illustrating cohort demographics,
APOE-ε4 allele enrichment, genomic PCA separation, and GWAS association
profiles across the Alzheimer's Disease Neuroimaging Initiative (ADNI) cohort.

Figures Generated:
1. results/figures/class_balance.png:
   Diagnostic breakdown of CN, MCI, and AD across the total cohort and batches.
2. results/figures/age_distribution.png:
   Age and education distributions by diagnosis category.
3. results/figures/apoe4_correlation.png:
   APOE-ε4 risk allele carrier rates and genotype breakdowns.
4. results/figures/genotype_pca.png:
   2D Principal Component Analysis on top selected genomic loci.
5. results/figures/top_snps_manhattan.png:
   Manhattan-style association ranking highlighting Chromosome 19 peak hits.

Author: Raja Omar
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("AEGIS-ND.EDA")

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
FIGURES_DIR = BASE_DIR / "results" / "figures"

# Professional publication styling
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.titlesize": 14,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

# Canonical clinical color palette
PALETTE = {
    "CN": "#2b5c8f",  # Deep slate blue
    "MCI": "#d9822b",  # Amber orange
    "AD": "#c0392b",  # Crimson red
    "Cohort 1": "#34495e",
    "Cohort 2": "#16a085",
}


def load_eda_data() -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, Dict]:
    """Load processed clinical records, binary cohort, feature matrices, and metadata."""
    clinical_csv = PROCESSED_DIR / "clinical_metadata.csv"
    binary_csv = PROCESSED_DIR / "binary_cohort_metadata.csv"
    xtrain_path = PROCESSED_DIR / "X_train.npy"
    xtest_path = PROCESSED_DIR / "X_test.npy"
    ytrain_path = PROCESSED_DIR / "y_train.npy"
    ytest_path = PROCESSED_DIR / "y_test.npy"
    features_json = PROCESSED_DIR / "selected_features.json"

    if not clinical_csv.exists() or not features_json.exists():
        raise FileNotFoundError(
            "Phase 2/3 artifacts not found. Please run data_loader.py and preprocessing.py first."
        )

    clinical_df = pd.read_csv(clinical_csv, index_col="PTID")
    binary_df = pd.read_csv(binary_csv, index_col="PTID")

    X_train = np.load(xtrain_path)
    X_test = np.load(xtest_path)
    y_train = np.load(ytrain_path)
    y_test = np.load(ytest_path)

    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])

    with open(features_json, "r", encoding="utf-8") as f:
        selected_meta = json.load(f)

    return clinical_df, binary_df, X_all, y_all, selected_meta


def plot_class_balance(clinical_df: pd.DataFrame, output_path: Path) -> None:
    """Plot distribution of diagnostic categories across the full ADNI cohort and sequencing batches."""
    logger.info("Generating Class Balance figure...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # 1. Total Diagnostic Distribution
    dx_counts = clinical_df["diagnosis"].value_counts()[["CN", "MCI", "AD"]]
    dx_pcts = (dx_counts / len(clinical_df)) * 100

    bars = axes[0].bar(
        dx_counts.index,
        dx_counts.values,
        color=[PALETTE["CN"], PALETTE["MCI"], PALETTE["AD"]],
        edgecolor="black",
        linewidth=1.2,
        width=0.55,
        alpha=0.9,
    )
    for bar, count, pct in zip(bars, dx_counts.values, dx_pcts.values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 10,
            f"N = {count}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )

    axes[0].set_title("Total Genotyped Cohort by Diagnosis (N = 793)", pad=15)
    axes[0].set_ylabel("Number of Participants")
    axes[0].set_xlabel("Clinical Diagnosis at Baseline")
    axes[0].set_ylim(0, max(dx_counts.values) * 1.22)
    axes[0].grid(axis="y", linestyle="--", alpha=0.7)

    # 2. Batch Harmonization Breakdown
    cohort_map = {
        "ADNI_GO_2_OmniExpress": "Cohort 1 (OmniExpress)",
        "ADNI_GO2_2nd": "Cohort 2 (2nd Orig)",
    }
    batch_df = clinical_df.copy()
    batch_df["cohort_clean"] = batch_df["cohort"].map(cohort_map)

    batch_ct = pd.crosstab(
        batch_df["cohort_clean"], batch_df["diagnosis"]
    )[["CN", "MCI", "AD"]]

    batch_ct.plot(
        kind="bar",
        stacked=False,
        ax=axes[1],
        color=[PALETTE["CN"], PALETTE["MCI"], PALETTE["AD"]],
        edgecolor="black",
        linewidth=1.1,
        width=0.65,
        alpha=0.9,
    )

    for p in axes[1].patches:
        h = p.get_height()
        if h > 0:
            axes[1].annotate(
                f"{int(h)}",
                (p.get_x() + p.get_width() / 2, h + 5),
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="semibold",
            )

    axes[1].set_title("Diagnostic Balance Across Sequencing Batches", pad=15)
    axes[1].set_ylabel("Number of Participants")
    axes[1].set_xlabel("Genotyping Array Batch")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
    axes[1].set_ylim(0, 260)
    axes[1].legend(title="Diagnosis", frameon=True, facecolor="white")
    axes[1].grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def plot_age_education_distribution(clinical_df: pd.DataFrame, output_path: Path) -> None:
    """Plot age and educational attainment distributions across clinical stages."""
    logger.info("Generating Age & Education Distribution figure...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    order = ["CN", "MCI", "AD"]

    # 1. Age boxplot with individual jitter points
    sns.boxplot(
        data=clinical_df,
        x="diagnosis",
        y="age",
        order=order,
        palette=[PALETTE["CN"], PALETTE["MCI"], PALETTE["AD"]],
        ax=axes[0],
        width=0.45,
        fliersize=0,
        boxprops=dict(alpha=0.85, edgecolor="black", linewidth=1.2),
        medianprops=dict(color="black", linewidth=2.0),
    )
    sns.stripplot(
        data=clinical_df,
        x="diagnosis",
        y="age",
        order=order,
        color="black",
        alpha=0.35,
        jitter=0.2,
        size=4.5,
        ax=axes[0],
    )

    # Annotate mean and std
    for i, dx in enumerate(order):
        sub_age = clinical_df[clinical_df["diagnosis"] == dx]["age"].dropna()
        axes[0].text(
            i,
            sub_age.min() - 4.5,
            f"μ = {sub_age.mean():.1f}y\nσ = {sub_age.std():.1f}",
            ha="center",
            va="top",
            fontsize=10.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9),
        )

    axes[0].set_title("Age Distribution at Baseline Exam", pad=15)
    axes[0].set_xlabel("Clinical Diagnosis")
    axes[0].set_ylabel("Age (Years)")
    axes[0].set_ylim(45, 98)

    # 2. Education violin / boxplot
    sns.violinplot(
        data=clinical_df,
        x="diagnosis",
        y="education",
        order=order,
        palette=[PALETTE["CN"], PALETTE["MCI"], PALETTE["AD"]],
        ax=axes[1],
        inner="quartile",
        cut=0,
        linewidth=1.2,
    )

    for i, dx in enumerate(order):
        sub_edu = clinical_df[clinical_df["diagnosis"] == dx]["education"].dropna()
        axes[1].text(
            i,
            21.5,
            f"μ = {sub_edu.mean():.1f} yrs",
            ha="center",
            va="bottom",
            fontweight="semibold",
            fontsize=10.5,
        )

    axes[1].set_title("Educational Attainment Across Groups", pad=15)
    axes[1].set_xlabel("Clinical Diagnosis")
    axes[1].set_ylabel("Completed Education (Years)")
    axes[1].set_ylim(4, 25)

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def plot_apoe4_correlation(clinical_df: pd.DataFrame, output_path: Path) -> None:
    """Plot APOE-ε4 carrier enrichment and allelic distribution across clinical categories."""
    logger.info("Generating APOE-ε4 Correlation figure...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    order = ["CN", "MCI", "AD"]

    # 1. Percentage of APOE-ε4 Carriers
    carrier_pct = []
    for dx in order:
        sub = clinical_df[clinical_df["diagnosis"] == dx]
        pct = (sub["apoe4_carrier"] == 1).mean() * 100.0
        carrier_pct.append(pct)

    bars = axes[0].bar(
        order,
        carrier_pct,
        color=[PALETTE["CN"], PALETTE["MCI"], PALETTE["AD"]],
        edgecolor="black",
        linewidth=1.2,
        width=0.5,
        alpha=0.9,
    )

    for bar, pct in zip(bars, carrier_pct):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2.0,
            f"{pct:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=12,
        )

    # Highlight risk ratio
    ad_cn_ratio = carrier_pct[2] / carrier_pct[0]
    axes[0].text(
        0.5,
        0.88,
        f"AD vs. CN Relative Risk Enrichment:\n{ad_cn_ratio:.2f}x Higher ε4 Frequency in AD",
        transform=axes[0].transAxes,
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff3cd", edgecolor="#ffeeba"),
    )

    axes[0].set_title("APOE-ε4 Allele Carrier Penetrance Rate (%)", pad=15)
    axes[0].set_ylabel("% Participants with ≥1 ε4 Allele")
    axes[0].set_xlabel("Clinical Diagnosis")
    axes[0].set_ylim(0, 85)
    axes[0].grid(axis="y", linestyle="--", alpha=0.7)

    # 2. Detailed Genotype Distribution (Stacked percentage)
    geno_order = ["2/2", "2/3", "2/4", "3/3", "3/4", "4/4"]
    geno_ct = pd.crosstab(
        clinical_df["diagnosis"], clinical_df["apoe_genotype"], normalize="index"
    ) * 100.0

    # Ensure all columns exist
    for g in geno_order:
        if g not in geno_ct.columns:
            geno_ct[g] = 0.0
    geno_ct = geno_ct.loc[order, geno_order]

    geno_palette = [
        "#a1d99b",  # 2/2 (protective green)
        "#74c476",  # 2/3 (protective)
        "#feb24c",  # 2/4 (neutral/risk)
        "#9ecae1",  # 3/3 (neutral blue)
        "#fc9272",  # 3/4 (high risk)
        "#de2d26",  # 4/4 (very high risk red)
    ]

    geno_ct.plot(
        kind="bar",
        stacked=True,
        ax=axes[1],
        color=geno_palette,
        edgecolor="black",
        linewidth=0.8,
        width=0.55,
    )

    axes[1].set_title("Full APOE Genotype Architecture by Diagnosis", pad=15)
    axes[1].set_ylabel("Proportion of Group (%)")
    axes[1].set_xlabel("Clinical Diagnosis")
    axes[1].set_xticklabels(order, rotation=0)
    axes[1].set_ylim(0, 100)
    axes[1].legend(
        title="APOE Genotype",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=True,
        facecolor="white",
    )
    axes[1].grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def plot_genotype_pca(
    X_all: np.ndarray,
    y_all: np.ndarray,
    output_path: Path,
) -> None:
    """Perform PCA on the 100 selected genomic variants and project into 2D space."""
    logger.info("Generating Genotype PCA figure...")
    # Extract only the 100 SNP dosages (columns 0 to 99)
    X_snps = X_all[:, :100]

    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(X_snps)

    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100

    pca_df = pd.DataFrame(
        {
            "PC1": pcs[:, 0],
            "PC2": pcs[:, 1],
            "Diagnosis": ["Alzheimer's Disease (AD)" if y == 1 else "Cognitively Normal (CN)" for y in y_all],
        }
    )

    fig, ax = plt.subplots(figsize=(9, 6.5))

    sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        hue="Diagnosis",
        palette={"Cognitively Normal (CN)": PALETTE["CN"], "Alzheimer's Disease (AD)": PALETTE["AD"]},
        style="Diagnosis",
        markers={"Cognitively Normal (CN)": "o", "Alzheimer's Disease (AD)": "X"},
        alpha=0.8,
        s=70,
        edgecolor="black",
        linewidth=0.5,
        ax=ax,
    )

    # Plot 2D Gaussian density contour centroids
    for diag, col in [("Cognitively Normal (CN)", PALETTE["CN"]), ("Alzheimer's Disease (AD)", PALETTE["AD"])]:
        sub = pca_df[pca_df["Diagnosis"] == diag]
        mean_pc1 = sub["PC1"].mean()
        mean_pc2 = sub["PC2"].mean()
        ax.scatter(
            mean_pc1,
            mean_pc2,
            color=col,
            edgecolor="white",
            s=220,
            linewidth=2.5,
            marker="*",
            zorder=10,
            label=f"{diag} Centroid",
        )

    ax.set_title(
        f"Principal Component Analysis of Top 100 Alzheimer's Loci (N = {len(y_all)})",
        pad=15,
    )
    ax.set_xlabel(f"Principal Component 1 ({var1:.1f}% Variance Explained)")
    ax.set_ylabel(f"Principal Component 2 ({var2:.1f}% Variance Explained)")
    ax.legend(frameon=True, facecolor="white", loc="upper right")

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def plot_manhattan_top_snps(selected_meta: Dict, output_path: Path) -> None:
    """Generate Manhattan-style association ranking plot highlighting Chromosome 19 hits."""
    logger.info("Generating Top SNPs Manhattan figure...")
    snps_data = selected_meta["top_snps"]
    df_snps = pd.DataFrame(snps_data)

    # Sort by chromosome then position
    df_snps = df_snps.sort_values(["chromosome", "position"]).reset_index(drop=True)
    df_snps["x_idx"] = np.arange(len(df_snps))

    fig, ax = plt.subplots(figsize=(14, 6))

    # Color alternating chromosomes, highlighting Chromosome 19 in red
    colors = []
    for c in df_snps["chromosome"]:
        if c == 19:
            colors.append("#c0392b")  # Bright red for Chr 19 APOE cluster
        elif c % 2 == 0:
            colors.append("#2b5c8f")  # Deep blue
        else:
            colors.append("#7f8c8d")  # Slate gray

    scatter = ax.scatter(
        df_snps["x_idx"],
        df_snps["welch_t_stat"],
        c=colors,
        s=df_snps["welch_t_stat"] * 18,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.7,
        zorder=3,
    )

    # Annotate peak chromosome 19 hits
    chr19_peaks = df_snps[
        (df_snps["chromosome"] == 19) & (df_snps["welch_t_stat"] >= 4.5)
    ].head(4)

    for _, row in chr19_peaks.iterrows():
        ax.annotate(
            f"{row['snp_id']}\n(t = {row['welch_t_stat']:.2f})",
            xy=(row["x_idx"], row["welch_t_stat"]),
            xytext=(row["x_idx"] + 2.5, row["welch_t_stat"] + 0.18),
            arrowprops=dict(
                facecolor="#c0392b",
                edgecolor="#c0392b",
                arrowstyle="->",
                lw=1.5,
            ),
            fontsize=9.5,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#fadbd8", edgecolor="#e74c3c", alpha=0.9),
            zorder=5,
        )

    # Draw genome-wide significance benchmark threshold line
    ax.axhline(4.0, color="#e67e22", linestyle="--", linewidth=1.2, label="Significance Threshold (|t| ≥ 4.0)")

    ax.set_title("Genome-Wide Association Screening: Top 100 Informative SNPs by Association Score", pad=15)
    ax.set_ylabel("Welch's Two-Sample |t|-Statistic")
    ax.set_xlabel("Selected SNPs Ordered by Genomic Coordinate (Chromosome 1 to 22)")
    ax.set_ylim(2.5, 5.8)

    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label="Chromosome 19 (APOE/TOMM40 Cluster)", markerfacecolor="#c0392b", markersize=10),
        Line2D([0], [0], marker="o", color="w", label="Autosomal Chromosomes (Even)", markerfacecolor="#2b5c8f", markersize=8),
        Line2D([0], [0], marker="o", color="w", label="Autosomal Chromosomes (Odd)", markerfacecolor="#7f8c8d", markersize=8),
        Line2D([0], [0], color="#e67e22", linestyle="--", label="Association Threshold (|t| ≥ 4.0)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", frameon=True, facecolor="white")

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def generate_all_figures() -> Dict[str, str]:
    """Generate and save all 5 Phase 4 figures to results/figures/."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    clinical_df, binary_df, X_all, y_all, selected_meta = load_eda_data()

    paths = {
        "class_balance": str(FIGURES_DIR / "class_balance.png"),
        "age_distribution": str(FIGURES_DIR / "age_distribution.png"),
        "apoe4_correlation": str(FIGURES_DIR / "apoe4_correlation.png"),
        "genotype_pca": str(FIGURES_DIR / "genotype_pca.png"),
        "top_snps_manhattan": str(FIGURES_DIR / "top_snps_manhattan.png"),
    }

    plot_class_balance(clinical_df, Path(paths["class_balance"]))
    plot_age_education_distribution(clinical_df, Path(paths["age_distribution"]))
    plot_apoe4_correlation(clinical_df, Path(paths["apoe4_correlation"]))
    plot_genotype_pca(X_all, y_all, Path(paths["genotype_pca"]))
    plot_manhattan_top_snps(selected_meta, Path(paths["top_snps_manhattan"]))

    logger.info("All 5 publication figures successfully generated in %s!", FIGURES_DIR)
    return paths


if __name__ == "__main__":
    generate_all_figures()
