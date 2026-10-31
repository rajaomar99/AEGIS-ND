"""AEGIS-ND: Baseline Target Model Training & Cross-Validation Pipeline (Phase 5)
=============================================================================
Trains predictive disease-risk classifiers for late-onset Alzheimer's Disease (AD)
using multi-modal genomic features (Top 100 SNPs + 4 Clinical Covariates).

Methodological Framework:
- Multi-Model Benchmarking:
  1. Primary Target Model: Random Forest Classifier (120 estimators, max_depth=4)
  2. Sequential Boosting: XGBoost Classifier (80 estimators, max_depth=3, lr=0.03)
  3. Regularized Linear Baseline: Logistic Regression (L2 penalty, C=0.1, lbfgs)
- 5-Fold Stratified Cross-Validation on training cohort (N=306) to measure internal stability.
- Out-of-Sample Generalization: Strict evaluation on the held-out test cohort (N=77).
- Generalization Gap Analysis: Quantifies Delta_gen = Train_AUC - Test_AUC,
  which establishes the empirical attack surface for Membership Inference Attacks.
- Artifact Export: Serializes primary target model to saved_models/target_model.pkl,
  exports metadata JSON, comparison tables, and publication-grade evaluation plots.

Author: Raja Omar
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from xgboost import XGBClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("AEGIS-ND.TargetModel")

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
SAVED_MODELS_DIR = BASE_DIR / "saved_models"
FIGURES_DIR = BASE_DIR / "results" / "figures"
TABLES_DIR = BASE_DIR / "results" / "tables"

# Visual styling
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "figure.titlesize": 14,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

MODEL_PALETTE = {
    "Random Forest (Primary Target)": "#2b5c8f",  # Deep Slate Blue
    "XGBoost Classifier": "#27ae60",               # Forest Green
    "Logistic Regression (L2)": "#8e44ad",         # Violet Purple
    "Chance": "#7f8c8d",                           # Neutral Gray
}


def load_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], Dict]:
    """Load preprocessed feature matrices, ground-truth labels, and feature metadata."""
    x_train_path = PROCESSED_DIR / "X_train.npy"
    x_test_path = PROCESSED_DIR / "X_test.npy"
    y_train_path = PROCESSED_DIR / "y_train.npy"
    y_test_path = PROCESSED_DIR / "y_test.npy"
    feat_json_path = PROCESSED_DIR / "selected_features.json"

    if not x_train_path.exists() or not feat_json_path.exists():
        raise FileNotFoundError(
            "Phase 3 artifacts not found. Please run src/preprocessing.py first."
        )

    X_train = np.load(x_train_path)
    X_test = np.load(x_test_path)
    y_train = np.load(y_train_path)
    y_test = np.load(y_test_path)

    with open(feat_json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Reconstruct feature names: 100 SNPs + 4 clinical covariates
    feature_names = [s["snp_id"] for s in meta["top_snps"]] + meta["clinical_covariates"]

    logger.info(
        "Loaded dataset: X_train %s, X_test %s, Total features: %d",
        X_train.shape,
        X_test.shape,
        len(feature_names),
    )
    return X_train, X_test, y_train, y_test, feature_names, meta


def get_model_specifications() -> Dict[str, Any]:
    """Instantiate model architectures with tuned hyperparameters."""
    return {
        "Random Forest (Primary Target)": RandomForestClassifier(
            n_estimators=120,
            max_depth=4,
            min_samples_leaf=3,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost Classifier": XGBClassifier(
            n_estimators=80,
            max_depth=3,
            learning_rate=0.03,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        ),
        "Logistic Regression (L2)": LogisticRegression(
            penalty="l2",
            C=0.1,
            solver="lbfgs",
            max_iter=1000,
            random_state=42,
        ),
    }


def evaluate_models_cross_validation(
    models: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_splits: int = 5,
) -> Dict[str, Dict[str, float]]:
    """Run Stratified K-Fold Cross-Validation on the training partition."""
    logger.info("Executing %d-Fold Stratified Cross-Validation on training data...", n_splits)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scoring = ["roc_auc", "accuracy", "f1", "precision", "recall"]

    cv_results = {}
    for name, model in models.items():
        res = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        cv_results[name] = {
            "cv_auc_mean": float(np.mean(res["test_roc_auc"])),
            "cv_auc_std": float(np.std(res["test_roc_auc"])),
            "cv_acc_mean": float(np.mean(res["test_accuracy"])),
            "cv_acc_std": float(np.std(res["test_accuracy"])),
            "cv_f1_mean": float(np.mean(res["test_f1"])),
            "cv_f1_std": float(np.std(res["test_f1"])),
            "cv_precision_mean": float(np.mean(res["test_precision"])),
            "cv_recall_mean": float(np.mean(res["test_recall"])),
        }
        logger.info(
            "  - %s: 5-Fold CV AUC = %.4f +/- %.4f | Accuracy = %.4f",
            name,
            cv_results[name]["cv_auc_mean"],
            cv_results[name]["cv_auc_std"],
            cv_results[name]["cv_acc_mean"],
        )

    return cv_results


def fit_and_evaluate_models(
    models: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cv_results: Dict[str, Dict[str, float]],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Fit all models, evaluate on held-out test cohort, and compute test metrics."""
    logger.info("Training models and evaluating on held-out test cohort (N=%d)...", len(y_test))

    records = []
    fitted_models = {}

    for name, model in models.items():
        # Fit on full training partition
        model.fit(X_train, y_train)
        fitted_models[name] = model

        # Probabilities and predictions
        y_train_proba = model.predict_proba(X_train)[:, 1]
        y_test_proba = model.predict_proba(X_test)[:, 1]
        y_test_pred = (y_test_proba >= 0.5).astype(int)

        # Performance metrics
        train_auc = float(roc_auc_score(y_train, y_train_proba))
        test_auc = float(roc_auc_score(y_test, y_test_proba))
        test_pr_auc = float(average_precision_score(y_test, y_test_proba))
        test_acc = float(accuracy_score(y_test, y_test_pred))
        test_f1 = float(f1_score(y_test, y_test_pred, zero_division=0))
        test_prec = float(precision_score(y_test, y_test_pred, zero_division=0))
        test_rec = float(recall_score(y_test, y_test_pred, zero_division=0))
        test_brier = float(brier_score_loss(y_test, y_test_proba))

        # Generalization gap (Train AUC - Test AUC)
        gen_gap = float(train_auc - test_auc)

        cv_info = cv_results[name]

        records.append(
            {
                "Model Architecture": name,
                "5-Fold CV ROC-AUC": f"{cv_info['cv_auc_mean']:.4f} ± {cv_info['cv_auc_std']:.4f}",
                "CV AUC Mean": cv_info["cv_auc_mean"],
                "Train ROC-AUC": round(train_auc, 4),
                "Test ROC-AUC": round(test_auc, 4),
                "Generalization Gap (Δ)": round(gen_gap, 4),
                "Test PR-AUC": round(test_pr_auc, 4),
                "Test Accuracy": round(test_acc, 4),
                "Test F1-Score": round(test_f1, 4),
                "Test Precision": round(test_prec, 4),
                "Test Recall": round(test_rec, 4),
                "Brier Score": round(test_brier, 4),
            }
        )

        logger.info(
            "  [%s] Train AUC: %.4f | Test AUC: %.4f | Δ_gen: %.4f | Test Acc: %.4f | Test F1: %.4f",
            name,
            train_auc,
            test_auc,
            gen_gap,
            test_acc,
            test_f1,
        )

    comparison_df = pd.DataFrame(records)
    return comparison_df, fitted_models


def plot_roc_curves(
    fitted_models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_path: Path,
) -> None:
    """Generate multi-model Receiver Operating Characteristic (ROC) curves."""
    logger.info("Generating multi-model ROC Curve comparison...")
    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    for name, model in fitted_models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)

        line_style = "-" if "Primary" in name else "--"
        line_width = 2.5 if "Primary" in name else 1.8
        ax.plot(
            fpr,
            tpr,
            label=f"{name} (AUC = {auc:.4f})",
            color=MODEL_PALETTE.get(name, "#34495e"),
            linestyle=line_style,
            linewidth=line_width,
        )

    # Reference diagonal (random chance)
    ax.plot([0, 1], [0, 1], linestyle=":", color=MODEL_PALETTE["Chance"], linewidth=1.5, label="Random Chance (AUC = 0.5000)")

    ax.set_title("Test Cohort ROC Performance (AD vs. CN, N = 77)", pad=15)
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.legend(loc="lower right", frameon=True, facecolor="white", shadow=True)
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Saved ROC Curves figure: %s", output_path)


def plot_confusion_matrix_figure(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_path: Path,
) -> None:
    """Plot normalized and raw confusion matrix for the primary target model."""
    logger.info("Generating Confusion Matrix figure...")
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    annot = np.array(
        [[f"{count}\n({norm*100:.1f}%)" for count, norm in zip(row_c, row_n)] for row_c, row_n in zip(cm, cm_norm)]
    )

    sns.heatmap(
        cm_norm,
        annot=annot,
        fmt="",
        cmap="Blues",
        cbar=True,
        xticklabels=["Normal (CN)", "Dementia (AD)"],
        yticklabels=["Normal (CN)", "Dementia (AD)"],
        linewidths=1.2,
        linecolor="black",
        ax=ax,
    )

    ax.set_title("Primary Target Model (Random Forest)\nTest Cohort Confusion Matrix (N = 77)", pad=15)
    ax.set_xlabel("Predicted Diagnosis")
    ax.set_ylabel("True Clinical Diagnosis")

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Saved Confusion Matrix figure: %s", output_path)


def plot_feature_importances(
    rf_model: RandomForestClassifier,
    feature_names: List[str],
    output_path: Path,
    top_n: int = 15,
) -> None:
    """Plot top predictive features ranked by Mean Decrease in Impurity (Gini)."""
    logger.info("Generating Feature Importance plot (Top %d features)...", top_n)
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    top_features = [feature_names[i] for i in indices]
    top_scores = importances[indices]

    # Color code: red for Chr 19 / APOE, blue for other SNPs, orange for clinical
    colors = []
    for f in top_features:
        if "apoe" in f.lower() or f in ["rs769449", "rs4420638", "rs157582", "rs2075650"]:
            colors.append("#c0392b")  # Crimson red for APOE cluster
        elif f in ["age", "sex", "education"]:
            colors.append("#e67e22")  # Amber for clinical demographics
        else:
            colors.append("#2b5c8f")  # Slate blue for other SNPs

    fig, ax = plt.subplots(figsize=(10, 6.5))
    bars = ax.barh(range(top_n), top_scores[::-1], color=colors[::-1], edgecolor="black", linewidth=1.0)

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.001, bar.get_y() + bar.get_height() / 2, f"{w*100:.2f}%", va="center", fontsize=10, fontweight="bold")

    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features[::-1], fontweight="semibold")
    ax.set_xlabel("Feature Importance (Gini Impurity Reduction)")
    ax.set_title(f"Top {top_n} Informative Features in Primary Target Model", pad=15)
    ax.set_xlim(0, max(top_scores) * 1.18)

    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#c0392b", lw=8, label="Chr 19 APOE/TOMM40 Risk Cluster"),
        Line2D([0], [0], color="#e67e22", lw=8, label="Clinical Demographics"),
        Line2D([0], [0], color="#2b5c8f", lw=8, label="Other Polygenic SNPs"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", frameon=True, facecolor="white")

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Saved Feature Importances figure: %s", output_path)


def export_target_artifacts(
    primary_model: RandomForestClassifier,
    comparison_df: pd.DataFrame,
    feature_names: List[str],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    metadata: Dict,
) -> Dict[str, str]:
    """Serialize target model, save metadata JSON, and export comparison CSV."""
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Serialize Primary Target Model
    model_path = SAVED_MODELS_DIR / "target_model.pkl"
    joblib.dump(primary_model, model_path)
    logger.info("Saved primary target model: %s", model_path)

    # 2. Export Model Metadata JSON
    train_proba = primary_model.predict_proba(X_train)[:, 1]
    test_proba = primary_model.predict_proba(X_test)[:, 1]
    train_auc = float(roc_auc_score(y_train, train_proba))
    test_auc = float(roc_auc_score(y_test, test_proba))
    test_pred = (test_proba >= 0.5).astype(int)

    model_metadata = {
        "model_architecture": "RandomForestClassifier",
        "library": "scikit-learn",
        "hyperparameters": {
            "n_estimators": 120,
            "max_depth": 4,
            "min_samples_leaf": 3,
            "max_features": "sqrt",
            "random_state": 42,
        },
        "features": {
            "total_count": len(feature_names),
            "snp_count": 100,
            "clinical_count": 4,
            "feature_names": feature_names,
        },
        "performance_metrics": {
            "train_roc_auc": round(train_auc, 4),
            "test_roc_auc": round(test_auc, 4),
            "test_accuracy": round(float(accuracy_score(y_test, test_pred)), 4),
            "test_f1": round(float(f1_score(y_test, test_pred, zero_division=0)), 4),
            "test_pr_auc": round(float(average_precision_score(y_test, test_proba)), 4),
            "brier_score": round(float(brier_score_loss(y_test, test_proba)), 4),
        },
        "generalization_gap": round(float(train_auc - test_auc), 4),
    }

    meta_path = SAVED_MODELS_DIR / "target_model_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(model_metadata, f, indent=2)
    logger.info("Saved target model metadata: %s", meta_path)

    # 3. Export Comparison Table CSV
    table_path = TABLES_DIR / "baseline_model_comparison.csv"
    comparison_df.to_csv(table_path, index=False)
    logger.info("Saved model comparison table: %s", table_path)

    return {
        "model_path": str(model_path),
        "metadata_path": str(meta_path),
        "table_path": str(table_path),
    }


def run_pipeline() -> Dict[str, str]:
    """Execute end-to-end Phase 5 target model training and evaluation."""
    logger.info("Starting Phase 5 Target Model Training & Evaluation Pipeline...")

    # Load data
    X_train, X_test, y_train, y_test, feature_names, meta = load_dataset()

    # Define architectures
    models = get_model_specifications()

    # 5-Fold Stratified Cross-Validation
    cv_results = evaluate_models_cross_validation(models, X_train, y_train, n_splits=5)

    # Fit and evaluate on held-out test cohort
    comparison_df, fitted_models = fit_and_evaluate_models(
        models, X_train, y_train, X_test, y_test, cv_results
    )

    primary_rf = fitted_models["Random Forest (Primary Target)"]

    # Generate Figures
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    p_roc = FIGURES_DIR / "baseline_roc_curve.png"
    p_cm = FIGURES_DIR / "baseline_confusion_matrix.png"
    p_imp = FIGURES_DIR / "feature_importances.png"

    plot_roc_curves(fitted_models, X_test, y_test, p_roc)
    plot_confusion_matrix_figure(primary_rf, X_test, y_test, p_cm)
    plot_feature_importances(primary_rf, feature_names, p_imp)

    # Export Model Artifacts
    artifacts = export_target_artifacts(
        primary_rf, comparison_df, feature_names, X_train, y_train, X_test, y_test, meta
    )
    artifacts["roc_curve"] = str(p_roc)
    artifacts["confusion_matrix"] = str(p_cm)
    artifacts["feature_importances"] = str(p_imp)

    logger.info("Phase 5 pipeline execution completed successfully!")
    return artifacts


if __name__ == "__main__":
    run_pipeline()
