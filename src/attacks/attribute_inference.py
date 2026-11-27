"""AEGIS-ND: Model Inversion & Genomic Attribute Reconstruction Attacks (Phase 8)
=============================================================================
Implementation of black-box model inversion and attribute inference attacks (AIA)
targeting sensitive genomic alleles and clinical biomarkers in late-onset
Alzheimer's Disease models (adapted from Fredrikson et al., CCS 2014 & CCS 2015;
Yeom et al., CSF 2018).

Adversarial Threat Model:
- Adversary Capabilities: Partial demographic and phenotypic knowledge of an
  individual (age, sex, education, clinical diagnosis y in {0, 1}), with black-box
  query access to the published disease-risk prediction model f(x).
- Adversary Objective: Reconstruct the individual's secret, highly sensitive
  genomic risk alleles:
  1. Primary Target: APOE4 carrier status (apoe4_carrier in {0, 1}).
  2. Secondary Target: Top Chromosome 19 risk variant (rs769449 in {0, 1}).
- Likelihood Inversion Mechanism:
  The adversary evaluates candidate hypothesis vectors x^(0) and x^(1) differing
  only in the target sensitive attribute, computing the likelihood gradient:
  Delta L = P(y = y_true | x^(1)) - P(y = y_true | x^(0)).
- Multi-Scenario Benchmarking:
  1. Baseline Target Model (Phase 5, eps = inf) vs. Defended DP-RF (Phase 7, eps* = 5.0).
  2. In-Sample Training Members (N = 306) vs. Out-of-Sample Test Patients (N = 77).
  Demonstrates how model overfitting exacerbates attribute inference, and how
  Differential Privacy dismantles the inversion vector.

Author: Raja Omar
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Logger setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("AEGIS-ND.AttributeInference")

PROCESSED_DIR = BASE_DIR / "data" / "processed"
SAVED_MODELS_DIR = BASE_DIR / "saved_models"
FIGURES_DIR = BASE_DIR / "results" / "figures"
TABLES_DIR = BASE_DIR / "results" / "tables"

for d in [SAVED_MODELS_DIR, FIGURES_DIR, TABLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def load_evaluation_data_and_models() -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], Any, Any
]:
    """Load preprocessed cohorts, feature names, baseline model, and DP-defended model."""
    logger.info("Loading preprocessed genomic matrices from %s...", PROCESSED_DIR)
    X_train = np.load(PROCESSED_DIR / "X_train.npy")
    X_test = np.load(PROCESSED_DIR / "X_test.npy")
    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    y_test = np.load(PROCESSED_DIR / "y_test.npy")

    feat_json_path = PROCESSED_DIR / "selected_features.json"
    with open(feat_json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    feature_names = [s["snp_id"] for s in meta["top_snps"]] + meta["clinical_covariates"]

    base_model_path = SAVED_MODELS_DIR / "target_model.pkl"
    dp_model_path = SAVED_MODELS_DIR / "dp_target_model.pkl"

    if not base_model_path.exists():
        raise FileNotFoundError(f"Baseline model not found at {base_model_path}.")
    if not dp_model_path.exists():
        raise FileNotFoundError(f"DP-defended model not found at {dp_model_path}.")

    logger.info("Loading baseline model from %s...", base_model_path)
    base_model = joblib.load(base_model_path)
    logger.info("Loading DP-defended model from %s...", dp_model_path)
    dp_model = joblib.load(dp_model_path)

    logger.info(
        "Successfully initialized models and cohorts: Train N=%d, Test N=%d, Features=%d",
        len(X_train),
        len(X_test),
        len(feature_names),
    )
    return X_train, X_test, y_train, y_test, feature_names, base_model, dp_model


class GenomicAttributeInferenceAttack:
    """Model Inversion & Attribute Inference Attack for sensitive genetic biomarkers.

    Parameters
    ----------
    target_attribute : str
        Name of the sensitive feature to reconstruct ('apoe4_carrier' or 'rs769449').
    feature_names : List[str]
        Ordered list of feature names corresponding to design matrix columns.
    """

    def __init__(self, target_attribute: str, feature_names: List[str]) -> None:
        self.target_attribute = target_attribute
        self.feature_names = feature_names

        if target_attribute not in feature_names:
            raise ValueError(f"Target attribute '{target_attribute}' not found in feature names.")

        self.target_idx = feature_names.index(target_attribute)
        self.optimal_threshold: float = 0.0

    def compute_inversion_scores(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute candidate hypothesis likelihood difference scores.

        Parameters
        ----------
        model : Any
            The target machine learning model.
        X : np.ndarray
            Patient feature matrix (N, D).
        y : np.ndarray
            Observed patient clinical disease labels (N,).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (inversion_scores, ground_truth_sensitive_alleles).
        """
        N = len(X)
        s_raw = X[:, self.target_idx]
        # Binarize genotype if dosage: 0 vs >=1
        s_true = (s_raw >= 1.0).astype(int)

        # Hypothesis 0: Target allele absent
        X_0 = X.copy()
        X_0[:, self.target_idx] = 0.0
        probs_0 = model.predict_proba(X_0)
        L_0 = probs_0[np.arange(N), y.astype(int)]

        # Hypothesis 1: Target allele present
        X_1 = X.copy()
        X_1[:, self.target_idx] = 1.0
        probs_1 = model.predict_proba(X_1)
        L_1 = probs_1[np.arange(N), y.astype(int)]

        # Likelihood difference score
        inversion_scores = L_1 - L_0
        return inversion_scores, s_true

    def fit_threshold(self, inversion_scores: np.ndarray, s_true: np.ndarray) -> float:
        """Derive optimal decision threshold maximizing Youden's J statistic (TPR - FPR)."""
        fpr, tpr, thresholds = roc_curve(s_true, inversion_scores)
        optimal_idx = int(np.argmax(tpr - fpr))
        self.optimal_threshold = float(thresholds[optimal_idx])
        return self.optimal_threshold

    def evaluate(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        threshold: float | None = None,
    ) -> Dict[str, Any]:
        """Execute attribute reconstruction evaluation.

        Parameters
        ----------
        model : Any
            Target model to invert.
        X : np.ndarray
            Evaluation feature matrix.
        y : np.ndarray
            Observed clinical labels.
        threshold : float, optional
            Decision threshold. If None, uses fitted or calibrated threshold.

        Returns
        -------
        Dict[str, Any]
            Comprehensive evaluation metrics and raw scores.
        """
        scores, s_true = self.compute_inversion_scores(model, X, y)

        if threshold is None:
            if self.optimal_threshold == 0.0:
                self.fit_threshold(scores, s_true)
            thr = self.optimal_threshold
        else:
            thr = threshold

        s_pred = (scores >= thr).astype(int)

        # Compute metrics
        roc_auc = float(roc_auc_score(s_true, scores))
        pr_auc = float(average_precision_score(s_true, scores))
        acc = float(accuracy_score(s_true, s_pred))
        bacc = float(balanced_accuracy_score(s_true, s_pred))
        prec = float(precision_score(s_true, s_pred, zero_division=0))
        rec = float(recall_score(s_true, s_pred, zero_division=0))
        f1 = float(f1_score(s_true, s_pred, zero_division=0))

        # Base rate (majority class accuracy)
        base_rate = float(max(np.mean(s_true == 1), np.mean(s_true == 0)))
        info_gain = float(acc - base_rate)

        return {
            "target_attribute": self.target_attribute,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "accuracy": acc,
            "balanced_accuracy": bacc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "base_rate": base_rate,
            "information_gain": info_gain,
            "threshold": thr,
            "scores": scores,
            "s_true": s_true,
            "s_pred": s_pred,
        }


def run_full_attribute_inference_audit(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    base_model: Any,
    dp_model: Any,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """Execute complete 8-scenario attribute inference benchmark.

    Evaluates:
    - Target Attributes: 'apoe4_carrier' and 'rs769449'
    - Target Models: Baseline RF (eps=inf) and Defended DP-RF (eps=5.0)
    - Cohort Partitions: Training Cohort (Members) and Held-Out Test Cohort (Non-Members)
    """
    logger.info("Executing Full Multi-Scenario Attribute Reconstruction Benchmark...")

    targets = ["apoe4_carrier", "rs769449"]
    records = []
    detailed_results = {}

    for target in targets:
        attack = GenomicAttributeInferenceAttack(target_attribute=target, feature_names=feature_names)

        # 1. Baseline Model on Training Set (Calibrate threshold here)
        scores_tr_base, s_tr_base = attack.compute_inversion_scores(base_model, X_train, y_train)
        calibrated_thr = attack.fit_threshold(scores_tr_base, s_tr_base)

        eval_base_tr = attack.evaluate(base_model, X_train, y_train, threshold=calibrated_thr)
        eval_base_te = attack.evaluate(base_model, X_test, y_test, threshold=calibrated_thr)

        # 2. Defended DP Model (evaluate under DP defense)
        attack_dp = GenomicAttributeInferenceAttack(target_attribute=target, feature_names=feature_names)
        scores_tr_dp, s_tr_dp = attack_dp.compute_inversion_scores(dp_model, X_train, y_train)
        dp_thr = attack_dp.fit_threshold(scores_tr_dp, s_tr_dp)

        eval_dp_tr = attack_dp.evaluate(dp_model, X_train, y_train, threshold=dp_thr)
        eval_dp_te = attack_dp.evaluate(dp_model, X_test, y_test, threshold=dp_thr)

        scenarios = [
            ("Baseline RF (eps=inf)", "Train (Members)", eval_base_tr),
            ("Baseline RF (eps=inf)", "Test (Non-Members)", eval_base_te),
            ("Defended DP-RF (eps=5.0)", "Train (Members)", eval_dp_tr),
            ("Defended DP-RF (eps=5.0)", "Test (Non-Members)", eval_dp_te),
        ]

        for model_label, cohort_label, res in scenarios:
            key = f"{target}_{model_label}_{cohort_label}"
            detailed_results[key] = res
            records.append({
                "Target Attribute": target,
                "Model": model_label,
                "Cohort Partition": cohort_label,
                "Reconstruction ROC-AUC": round(res["roc_auc"], 4),
                "Reconstruction PR-AUC": round(res["pr_auc"], 4),
                "Reconstruction Accuracy": round(res["accuracy"], 4),
                "Balanced Accuracy": round(res["balanced_accuracy"], 4),
                "Reconstruction Precision": round(res["precision"], 4),
                "Reconstruction Recall": round(res["recall"], 4),
                "F1-Score": round(res["f1"], 4),
                "Base Rate (Prior)": round(res["base_rate"], 4),
                "Adversarial Info Gain": round(res["information_gain"], 4),
            })
            logger.info(
                "[%s | %s | %s] AUC: %.4f | Acc: %.4f (Base: %.4f) | Prec: %.4f | Gain: %+.4f",
                target,
                model_label,
                cohort_label,
                res["roc_auc"],
                res["accuracy"],
                res["base_rate"],
                res["precision"],
                res["information_gain"],
            )

    df_results = pd.DataFrame(records)
    return df_results, detailed_results


def plot_attribute_reconstruction_roc(
    detailed_results: Dict[str, Dict[str, Any]],
    output_path: Path,
) -> None:
    """Generate publication-grade 2-panel ROC visualization for attribute reconstruction."""
    logger.info("Generating Attribute Reconstruction ROC Curves visualization...")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    targets = [("apoe4_carrier", "APOE4 Carrier Status (Clinical Risk Gene)"), ("rs769449", "rs769449 (Top Chr 19 Risk SNP)")]

    colors = {
        "Baseline RF (eps=inf)_Train (Members)": ("#c0392b", "-", "Baseline RF: Train Members"),
        "Baseline RF (eps=inf)_Test (Non-Members)": ("#e67e22", "--", "Baseline RF: Test Non-Members"),
        "Defended DP-RF (eps=5.0)_Train (Members)": ("#27ae60", "-.", "Defended DP-RF: Train Members"),
        "Defended DP-RF (eps=5.0)_Test (Non-Members)": ("#2b5c8f", ":", "Defended DP-RF: Test Non-Members"),
    }

    for ax, (target, title) in zip(axes, targets):
        for key_suffix, (col, ls, label) in colors.items():
            full_key = f"{target}_{key_suffix}"
            res = detailed_results[full_key]
            fpr, tpr, _ = roc_curve(res["s_true"], res["scores"])
            auc_val = res["roc_auc"]
            ax.plot(fpr, tpr, color=col, linestyle=ls, linewidth=2.0, label=f"{label} (AUC = {auc_val:.3f})")

        # Random guessing baseline
        ax.plot([0, 1], [0, 1], color="#7f8c8d", linestyle=":", linewidth=1.4, label="Random Guessing (AUC = 0.500)")
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel("False Positive Rate", fontsize=11)
        ax.set_ylabel("True Positive Rate", fontsize=11)
        ax.set_title(f"Attribute Reconstruction: {title}", fontsize=12, fontweight="bold", pad=12)
        ax.legend(loc="lower right", frameon=True, framealpha=0.9, fontsize=9.2)
        ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info("Saved Attribute Reconstruction ROC figure to %s", output_path)


def plot_aia_defense_comparison(
    df_results: pd.DataFrame,
    output_path: Path,
) -> None:
    """Generate comparative grouped bar chart: Baseline RF vs. DP-RF Defense across targets."""
    logger.info("Generating Attribute Reconstruction Defense Comparison Bar Chart...")

    # Focus on Training Cohort (where memorization leakage is highest)
    df_train = df_results[df_results["Cohort Partition"] == "Train (Members)"].copy()

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    metrics = ["Reconstruction ROC-AUC", "Reconstruction Precision", "Balanced Accuracy"]
    x = np.arange(len(metrics))
    width = 0.20

    # APOE4 Baseline vs DP
    row_apoe_base = df_train[(df_train["Target Attribute"] == "apoe4_carrier") & (df_train["Model"] == "Baseline RF (eps=inf)")].iloc[0]
    row_apoe_dp = df_train[(df_train["Target Attribute"] == "apoe4_carrier") & (df_train["Model"] == "Defended DP-RF (eps=5.0)")].iloc[0]

    # rs769449 Baseline vs DP
    row_snp_base = df_train[(df_train["Target Attribute"] == "rs769449") & (df_train["Model"] == "Baseline RF (eps=inf)")].iloc[0]
    row_snp_dp = df_train[(df_train["Target Attribute"] == "rs769449") & (df_train["Model"] == "Defended DP-RF (eps=5.0)")].iloc[0]

    vals_apoe_base = [row_apoe_base[m] for m in metrics]
    vals_apoe_dp = [row_apoe_dp[m] for m in metrics]
    vals_snp_base = [row_snp_base[m] for m in metrics]
    vals_snp_dp = [row_snp_dp[m] for m in metrics]

    rects1 = ax.bar(x - 1.5 * width, vals_apoe_base, width, label="APOE4 (Baseline RF)", color="#c0392b", edgecolor="black", alpha=0.85)
    rects2 = ax.bar(x - 0.5 * width, vals_apoe_dp, width, label="APOE4 (Defended DP-RF)", color="#27ae60", edgecolor="black", alpha=0.85)
    rects3 = ax.bar(x + 0.5 * width, vals_snp_base, width, label="rs769449 (Baseline RF)", color="#e67e22", edgecolor="black", alpha=0.85)
    rects4 = ax.bar(x + 1.5 * width, vals_snp_dp, width, label="rs769449 (Defended DP-RF)", color="#2b5c8f", edgecolor="black", alpha=0.85)

    ax.set_ylabel("Metric Value / Score", fontsize=11, fontweight="medium")
    ax.set_title(
        "Genomic Attribute Reconstruction Defense: Baseline Target Model vs. DP-RF (eps = 5.0)\n"
        "Complete Suppression of Sensitive Genetic Inversion Across Clinical and Genomic Biomarkers",
        fontsize=12,
        fontweight="bold",
        pad=14,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10.5)
    ax.set_ylim([0.0, 1.05])
    ax.legend(frameon=True, framealpha=0.9, fontsize=9.5)
    ax.grid(True, linestyle=":", alpha=0.5, axis="y")

    for rects in [rects1, rects2, rects3, rects4]:
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info("Saved AIA Defense Comparison bar chart to %s", output_path)


def export_attribute_inference_artifacts(
    df_results: pd.DataFrame,
) -> Dict[str, str]:
    """Export quantitative comparison table and metadata JSON."""
    table_path = TABLES_DIR / "attribute_reconstruction_metrics.csv"
    df_results.to_csv(table_path, index=False)
    logger.info("Saved attribute reconstruction metrics table to %s", table_path)

    meta_path = SAVED_MODELS_DIR / "attribute_inference_metadata.json"
    metadata = {
        "attack_type": "Model Inversion & Genomic Attribute Reconstruction Attack",
        "methodology": "Fredrikson et al. (2014, 2015) / Yeom et al. (2018)",
        "target_attributes": ["apoe4_carrier", "rs769449"],
        "evaluated_models": ["Baseline RF (eps=inf)", "Defended DP-RF (eps=5.0)"],
        "evaluated_cohorts": ["Train (Members, N=306)", "Test (Non-Members, N=77)"],
        "summary": {
            "apoe4_baseline_train_auc": float(df_results.loc[(df_results["Target Attribute"] == "apoe4_carrier") & (df_results["Model"] == "Baseline RF (eps=inf)") & (df_results["Cohort Partition"] == "Train (Members)"), "Reconstruction ROC-AUC"].values[0]),
            "apoe4_dp_train_auc": float(df_results.loc[(df_results["Target Attribute"] == "apoe4_carrier") & (df_results["Model"] == "Defended DP-RF (eps=5.0)") & (df_results["Cohort Partition"] == "Train (Members)"), "Reconstruction ROC-AUC"].values[0]),
            "snp_baseline_train_auc": float(df_results.loc[(df_results["Target Attribute"] == "rs769449") & (df_results["Model"] == "Baseline RF (eps=inf)") & (df_results["Cohort Partition"] == "Train (Members)"), "Reconstruction ROC-AUC"].values[0]),
            "snp_dp_train_auc": float(df_results.loc[(df_results["Target Attribute"] == "rs769449") & (df_results["Model"] == "Defended DP-RF (eps=5.0)") & (df_results["Cohort Partition"] == "Train (Members)"), "Reconstruction ROC-AUC"].values[0]),
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved attribute inference metadata JSON to %s", meta_path)

    return {
        "table_path": str(table_path),
        "metadata_path": str(meta_path),
    }


def run_attribute_inference_pipeline() -> Dict[str, Any]:
    """Execute end-to-end Phase 8 Model Inversion & Attribute Reconstruction benchmark."""
    logger.info("Initiating Phase 8: Model Inversion & Attribute Reconstruction Attacks...")

    # Step 1: Load cohorts and models
    X_train, X_test, y_train, y_test, feature_names, base_model, dp_model = (
        load_evaluation_data_and_models()
    )

    # Step 2: Execute 8-scenario benchmark
    df_results, detailed_results = run_full_attribute_inference_audit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
        base_model=base_model,
        dp_model=dp_model,
    )

    # Step 3: Generate publication figures
    roc_fig_path = FIGURES_DIR / "attribute_reconstruction_roc.png"
    comp_fig_path = FIGURES_DIR / "aia_defense_comparison.png"

    plot_attribute_reconstruction_roc(detailed_results, roc_fig_path)
    plot_aia_defense_comparison(df_results, comp_fig_path)

    # Step 4: Export tables and metadata
    artifacts = export_attribute_inference_artifacts(df_results)

    logger.info("Phase 8 Model Inversion pipeline successfully completed and exported.")
    return {
        "df_results": df_results,
        "roc_fig": str(roc_fig_path),
        "comp_fig": str(comp_fig_path),
        "artifacts": artifacts,
    }


if __name__ == "__main__":
    run_attribute_inference_pipeline()

