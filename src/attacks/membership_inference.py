"""AEGIS-ND: Shadow-Model Membership Inference Attack (MIA) Framework (Phase 6)
=============================================================================
Implementation of black-box shadow-model membership inference attacks against
genomic disease-risk prediction models, adapted from Shokri et al. (IEEE S&P 2017)
and Salem et al. (NDSS 2019).

Adversarial Threat Model:
- Black-Box Query Access: The adversary queries the target model with a genomic
  profile x* (104 features) and receives posterior class probabilities f_target(x*).
- Adversary Objective: Infer whether a specific patient (x*, y*) was a member
  of the target model's private training cohort (D_train, N=306) versus an
  unseen out-of-sample patient (D_test, N=77).
- Auxiliary Distribution: The adversary trains S shadow models on partitions of
  the auxiliary genomic data, mimicking the target architecture to construct
  a synthetic dataset of member (1) and non-member (0) confidence vectors.
- Information-Theoretic Confidence Signatures: Extracted 5D features:
  [P(y_true), max_c P(y_c), |P(1) - P(0)|, H(P), P(AD)].
- Attack Meta-Classifier: Calibrated regularized Logistic Regression classifier
  trained on standardized shadow confidence signatures.
- Cryptographic Advantage: Measures Adversarial Advantage = 2 * (AUC - 0.5)
  and Empirical Privacy Leakage = AUC - 0.5.

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
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

# Set up logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("AEGIS-ND.MIA")

# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
SAVED_MODELS_DIR = BASE_DIR / "saved_models"
FIGURES_DIR = BASE_DIR / "results" / "figures"
TABLES_DIR = BASE_DIR / "results" / "tables"

for d in [SAVED_MODELS_DIR, FIGURES_DIR, TABLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def load_genomic_data_and_target_model() -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], RandomForestClassifier
]:
    """Load preprocessed genomic matrices, labels, feature names, and trained target model."""
    logger.info("Loading preprocessed training and test partitions from %s...", PROCESSED_DIR)
    X_train = np.load(PROCESSED_DIR / "X_train.npy")
    X_test = np.load(PROCESSED_DIR / "X_test.npy")
    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    y_test = np.load(PROCESSED_DIR / "y_test.npy")

    with open(PROCESSED_DIR / "feature_names.json", "r", encoding="utf-8") as f:
        feature_names = json.load(f)

    target_model_path = SAVED_MODELS_DIR / "target_model.pkl"
    if not target_model_path.exists():
        raise FileNotFoundError(
            f"Trained target model not found at {target_model_path}. "
            "Please run Phase 5 (src/models/target_model.py) first."
        )

    logger.info("Loading serialized primary target model from %s...", target_model_path)
    target_model = joblib.load(target_model_path)

    logger.info(
        "Successfully loaded data: Train N=%d, Test N=%d, Features=%d, Model=%s",
        len(X_train),
        len(X_test),
        X_train.shape[1],
        target_model.__class__.__name__,
    )
    return X_train, X_test, y_train, y_test, feature_names, target_model


def extract_attack_features(probs: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Extract information-theoretic confidence signatures from model class probabilities.

    Given posterior probabilities P(y|x) and true class labels y_true, construct a
    5-dimensional feature vector v(x):
      1. P(y = y_true | x): Posterior probability assigned to the ground-truth diagnosis.
      2. max_c P(y = c | x): Peak prediction confidence across all classes.
      3. |P(1) - P(0)|: Prediction margin / confidence gap between classes.
      4. H(P): Shannon entropy measuring prediction uncertainty.
      5. P(AD | x): Raw posterior probability of the positive disease class (AD).

    Parameters
    ----------
    probs : np.ndarray
        Posterior probability array of shape (N, 2), where column 0 is P(CN)
        and column 1 is P(AD).
    y_true : np.ndarray
        Ground-truth clinical labels of shape (N,), with values in {0, 1}.

    Returns
    -------
    np.ndarray
        Extracted attack feature matrix of shape (N, 5).
    """
    eps = 1e-12  # Numerical stability epsilon for entropy calculation
    probs_clipped = np.clip(probs, eps, 1.0 - eps)

    # 1. Probability of true clinical label
    n_samples = len(y_true)
    p_true = probs[np.arange(n_samples), y_true.astype(int)]

    # 2. Maximum posterior probability
    p_max = np.max(probs, axis=1)

    # 3. Margin / Confidence gap
    margin = np.abs(probs[:, 1] - probs[:, 0])

    # 4. Shannon Entropy: H(P) = - sum_c P(c) * log2(P(c))
    entropy = -np.sum(probs_clipped * np.log2(probs_clipped), axis=1)

    # 5. Raw disease probability
    p_ad = probs[:, 1]

    features = np.column_stack([p_true, p_max, margin, entropy, p_ad])
    return features


FEATURE_NAMES_MIA = [
    "P(True Label)",
    "P(Max)",
    "Margin |P1 - P0|",
    "Shannon Entropy H(P)",
    "P(AD)",
]


class ShadowModelMIA:
    """Black-Box Shadow-Model Membership Inference Attack.

    Trains multiple shadow models mimicking the target architecture to construct
    a synthetic membership inference training dataset, then fits a calibrated
    meta-classifier to distinguish members from non-members based on confidence signatures.

    Parameters
    ----------
    n_shadow_models : int, default=10
        Number of shadow models to train on bootstrap/subsampled partitions.
    shadow_train_ratio : float, default=0.5
        Proportion of auxiliary data allocated to in-sample training for each shadow model.
    meta_classifier_C : float, default=1.0
        Inverse regularization strength for the Logistic Regression attack meta-classifier.
    random_state : int, default=42
        Master random seed for reproducibility.
    """

    def __init__(
        self,
        n_shadow_models: int = 10,
        shadow_train_ratio: float = 0.5,
        meta_classifier_C: float = 1.0,
        random_state: int = 42,
    ) -> None:
        self.n_shadow_models = n_shadow_models
        self.shadow_train_ratio = shadow_train_ratio
        self.meta_classifier_C = meta_classifier_C
        self.random_state = random_state

        self.shadow_models: List[RandomForestClassifier] = []
        self.scaler = StandardScaler()
        self.meta_classifier = LogisticRegression(
            penalty="l2",
            C=meta_classifier_C,
            solver="lbfgs",
            max_iter=1000,
            random_state=random_state,
        )
        self.is_fitted = False

    def fit(self, X_aux: np.ndarray, y_aux: np.ndarray) -> "ShadowModelMIA":
        """Train shadow models and fit the attack meta-classifier.

        Parameters
        ----------
        X_aux : np.ndarray
            Auxiliary feature matrix of shape (N, D).
        y_aux : np.ndarray
            Auxiliary disease labels of shape (N,).

        Returns
        -------
        ShadowModelMIA
            Fitted attack instance.
        """
        logger.info(
            "Training %d shadow models (split ratio = %.2f) on auxiliary cohort (N=%d)...",
            self.n_shadow_models,
            self.shadow_train_ratio,
            len(X_aux),
        )

        all_attack_features: List[np.ndarray] = []
        all_attack_labels: List[np.ndarray] = []

        sss = StratifiedShuffleSplit(
            n_splits=self.n_shadow_models,
            train_size=self.shadow_train_ratio,
            random_state=self.random_state,
        )

        for s_idx, (idx_in, idx_out) in enumerate(sss.split(X_aux, y_aux)):
            X_s_in, y_s_in = X_aux[idx_in], y_aux[idx_in]
            X_s_out, y_s_out = X_aux[idx_out], y_aux[idx_out]

            # Shadow model matches target architecture: RF with 120 trees, max_depth=4
            shadow_model = RandomForestClassifier(
                n_estimators=120,
                max_depth=4,
                min_samples_leaf=3,
                max_features="sqrt",
                random_state=self.random_state + s_idx * 101,
                n_jobs=-1,
            )
            shadow_model.fit(X_s_in, y_s_in)
            self.shadow_models.append(shadow_model)

            # In-sample predictions (Member, y_mia = 1)
            probs_in = shadow_model.predict_proba(X_s_in)
            feats_in = extract_attack_features(probs_in, y_s_in)
            labels_in = np.ones(len(X_s_in), dtype=int)

            # Out-of-sample predictions (Non-Member, y_mia = 0)
            probs_out = shadow_model.predict_proba(X_s_out)
            feats_out = extract_attack_features(probs_out, y_s_out)
            labels_out = np.zeros(len(X_s_out), dtype=int)

            all_attack_features.extend([feats_in, feats_out])
            all_attack_labels.extend([labels_in, labels_out])

        # Aggregate synthetic attack dataset
        X_attack = np.vstack(all_attack_features)
        y_attack = np.concatenate(all_attack_labels)

        logger.info(
            "Constructed shadow attack dataset: %d samples (%d members, %d non-members, 5 features).",
            len(X_attack),
            int(np.sum(y_attack == 1)),
            int(np.sum(y_attack == 0)),
        )

        # Standardize attack features and fit meta-classifier
        X_attack_scaled = self.scaler.fit_transform(X_attack)
        self.meta_classifier.fit(X_attack_scaled, y_attack)
        self.is_fitted = True

        cv_score = self.meta_classifier.score(X_attack_scaled, y_attack)
        logger.info("Attack meta-classifier trained. In-sample training accuracy = %.4f", cv_score)
        return self

    def predict_membership_proba(
        self,
        target_model: RandomForestClassifier,
        X: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """Infer membership probabilities for query samples against the target model.

        Parameters
        ----------
        target_model : RandomForestClassifier
            The black-box target model.
        X : np.ndarray
            Feature matrix of query individuals.
        y : np.ndarray
            True clinical labels of query individuals.

        Returns
        -------
        np.ndarray
            Array of membership probabilities P(Member = 1 | x).
        """
        if not self.is_fitted:
            raise RuntimeError("ShadowModelMIA must be fitted before running inference.")

        probs = target_model.predict_proba(X)
        features = extract_attack_features(probs, y)
        features_scaled = self.scaler.transform(features)
        return self.meta_classifier.predict_proba(features_scaled)[:, 1]

    def evaluate_on_target(
        self,
        target_model: RandomForestClassifier,
        X_members: np.ndarray,
        y_members: np.ndarray,
        X_non_members: np.ndarray,
        y_non_members: np.ndarray,
    ) -> Dict[str, Any]:
        """Evaluate the shadow-model attack against the real target model.

        Members: Real target model training cohort (N=306, y_mia = 1).
        Non-Members: Held-out test cohort (N=77, y_mia = 0).

        Parameters
        ----------
        target_model : RandomForestClassifier
            The trained target model.
        X_members : np.ndarray
            Training features (N_train = 306).
        y_members : np.ndarray
            Training labels.
        X_non_members : np.ndarray
            Held-out test features (N_test = 77).
        y_non_members : np.ndarray
            Held-out test labels.

        Returns
        -------
        Dict[str, Any]
            Comprehensive quantitative attack evaluation metrics and raw predictions.
        """
        logger.info("Evaluating Shadow-Model MIA against real target model...")
        scores_members = self.predict_membership_proba(target_model, X_members, y_members)
        scores_non_members = self.predict_membership_proba(target_model, X_non_members, y_non_members)

        y_eval_true = np.concatenate([np.ones(len(scores_members)), np.zeros(len(scores_non_members))])
        y_eval_scores = np.concatenate([scores_members, scores_non_members])
        y_eval_preds = (y_eval_scores >= 0.5).astype(int)

        roc_auc = float(roc_auc_score(y_eval_true, y_eval_scores))
        pr_auc = float(average_precision_score(y_eval_true, y_eval_scores))
        acc = float(accuracy_score(y_eval_true, y_eval_preds))
        prec = float(precision_score(y_eval_true, y_eval_preds, zero_division=0))
        rec = float(recall_score(y_eval_true, y_eval_preds, zero_division=0))
        f1 = float(f1_score(y_eval_true, y_eval_preds, zero_division=0))
        adv = float(2.0 * (roc_auc - 0.5))
        leakage = float(roc_auc - 0.5)

        logger.info("=== EMPIRICAL ATTACK RESULTS ON TARGET MODEL ===")
        logger.info("  - Attack ROC-AUC:              %.4f", roc_auc)
        logger.info("  - Attack PR-AUC:               %.4f", pr_auc)
        logger.info("  - Adversarial Advantage:       %.4f", adv)
        logger.info("  - Privacy Leakage (AUC - 0.5): %.4f", leakage)
        logger.info("  - Attack Accuracy (thr=0.5):   %.4f", acc)
        logger.info("  - Attack Precision:            %.4f", prec)
        logger.info("  - Attack Recall:               %.4f", rec)
        logger.info("  - Attack F1-Score:             %.4f", f1)
        logger.info("  - Mean Score (Members):        %.4f +/- %.4f", np.mean(scores_members), np.std(scores_members))
        logger.info("  - Mean Score (Non-Members):    %.4f +/- %.4f", np.mean(scores_non_members), np.std(scores_non_members))

        return {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "advantage": adv,
            "privacy_leakage": leakage,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "n_members": len(scores_members),
            "n_non_members": len(scores_non_members),
            "mean_member_score": float(np.mean(scores_members)),
            "std_member_score": float(np.std(scores_members)),
            "mean_non_member_score": float(np.mean(scores_non_members)),
            "std_non_member_score": float(np.std(scores_non_members)),
            "y_eval_true": y_eval_true,
            "y_eval_scores": y_eval_scores,
            "y_eval_preds": y_eval_preds,
            "scores_members": scores_members,
            "scores_non_members": scores_non_members,
        }


def plot_attack_roc_curve(
    y_eval_true: np.ndarray,
    y_eval_scores: np.ndarray,
    roc_auc: float,
    advantage: float,
    output_path: Path,
) -> None:
    """Plot publication-grade ROC curve of the Membership Inference Attack."""
    logger.info("Generating MIA ROC curve visualization...")
    fpr, tpr, _ = roc_curve(y_eval_true, y_eval_scores)

    fig, ax = plt.subplots(figsize=(7, 6))

    # Attack ROC curve
    ax.plot(
        fpr,
        tpr,
        color="#c0392b",
        linewidth=2.4,
        label=f"Shadow-Model MIA (AUC = {roc_auc:.4f})",
    )
    ax.fill_between(fpr, tpr, alpha=0.15, color="#c0392b")

    # Random baseline
    ax.plot(
        [0, 1],
        [0, 1],
        color="#7f8c8d",
        linestyle="--",
        linewidth=1.5,
        label="Random Guessing (AUC = 0.5000)",
    )

    # Annotations
    ax.annotate(
        f"Adversarial Advantage: {advantage:.4f}\n"
        f"Privacy Leakage: +{roc_auc - 0.5:.4f}\n"
        f"Target Cohort: N = 383\n"
        f"  - Members (Train): 306\n"
        f"  - Non-Members (Test): 77",
        xy=(0.48, 0.15),
        xycoords="axes fraction",
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor="#bdc3c7", alpha=0.9),
    )

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate (Non-Members misclassified as Members)", fontsize=11, fontweight="medium")
    ax.set_ylabel("True Positive Rate (Members correctly identified)", fontsize=11, fontweight="medium")
    ax.set_title(
        "Black-Box Membership Inference Attack on Baseline Target Model\n"
        "Receiver Operating Characteristic (ROC) Curve",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.legend(loc="lower right", frameon=True, framealpha=0.9, edgecolor="#bdc3c7", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info("Saved MIA ROC curve figure to %s", output_path)


def plot_attack_score_distribution(
    scores_members: np.ndarray,
    scores_non_members: np.ndarray,
    output_path: Path,
) -> None:
    """Plot KDE and density distribution of membership scores for members vs non-members."""
    logger.info("Generating MIA Score Distribution visualization...")
    fig, ax = plt.subplots(figsize=(8, 5.5))

    # Kernel Density Estimates
    sns.kdeplot(
        scores_members,
        ax=ax,
        color="#c0392b",
        fill=True,
        alpha=0.35,
        linewidth=2.2,
        label=f"Members (D_train, N={len(scores_members)})",
    )
    sns.kdeplot(
        scores_non_members,
        ax=ax,
        color="#2b5c8f",
        fill=True,
        alpha=0.35,
        linewidth=2.2,
        label=f"Non-Members (D_test, N={len(scores_non_members)})",
    )

    # Vertical reference lines for means
    mean_mem = np.mean(scores_members)
    mean_non = np.mean(scores_non_members)
    ax.axvline(
        mean_mem,
        color="#c0392b",
        linestyle="--",
        linewidth=1.6,
        label=f"Mean Member Score ({mean_mem:.3f})",
    )
    ax.axvline(
        mean_non,
        color="#2b5c8f",
        linestyle="--",
        linewidth=1.6,
        label=f"Mean Non-Member Score ({mean_non:.3f})",
    )

    # Decision threshold
    ax.axvline(
        0.5,
        color="#7f8c8d",
        linestyle=":",
        linewidth=1.4,
        label="Default Decision Threshold (0.50)",
    )

    ax.set_xlim([-0.05, 1.05])
    ax.set_xlabel("Predicted Membership Probability P(Member | x)", fontsize=11, fontweight="medium")
    ax.set_ylabel("Kernel Density", fontsize=11, fontweight="medium")
    ax.set_title(
        "MIA Score Distribution: Memorized Training Cohort vs. Unseen Test Patients\n"
        f"Separation Gap = {mean_mem - mean_non:+.3f} (Generalization Leakage)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.legend(loc="upper left", frameon=True, framealpha=0.9, edgecolor="#bdc3c7", fontsize=9.5)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info("Saved MIA score distribution figure to %s", output_path)


def export_attack_artifacts(
    eval_results: Dict[str, Any],
    attack_model: ShadowModelMIA,
) -> Dict[str, str]:
    """Export attack evaluation tables, metadata JSON, and serialized attack model."""
    # 1. Export CSV metrics table
    table_path = TABLES_DIR / "attack_metrics_baseline.csv"
    metrics_records = [
        {"Metric": "Attack ROC-AUC", "Value": f"{eval_results['roc_auc']:.4f}", "Interpretation": "Discrimination power of attack (0.5=random, 1.0=full leak)"},
        {"Metric": "Attack PR-AUC", "Value": f"{eval_results['pr_auc']:.4f}", "Interpretation": "Precision-recall area under curve accounting for class imbalance"},
        {"Metric": "Adversarial Advantage", "Value": f"{eval_results['advantage']:.4f}", "Interpretation": "Cryptographic advantage = 2 * (AUC - 0.5)"},
        {"Metric": "Empirical Privacy Leakage", "Value": f"{eval_results['privacy_leakage']:.4f}", "Interpretation": "AUC excess above random guessing benchmark"},
        {"Metric": "Attack Accuracy (thr=0.5)", "Value": f"{eval_results['accuracy']:.4f}", "Interpretation": "Classification accuracy on membership status"},
        {"Metric": "Attack Precision", "Value": f"{eval_results['precision']:.4f}", "Interpretation": "Fraction of predicted members who were true members"},
        {"Metric": "Attack Recall", "Value": f"{eval_results['recall']:.4f}", "Interpretation": "Fraction of true members successfully re-identified"},
        {"Metric": "Attack F1-Score", "Value": f"{eval_results['f1']:.4f}", "Interpretation": "Harmonic mean of attack precision and recall"},
        {"Metric": "Mean Score (Members)", "Value": f"{eval_results['mean_member_score']:.4f}", "Interpretation": "Average inferred membership probability for training subjects"},
        {"Metric": "Mean Score (Non-Members)", "Value": f"{eval_results['mean_non_member_score']:.4f}", "Interpretation": "Average inferred membership probability for test subjects"},
    ]
    df_metrics = pd.DataFrame(metrics_records)
    df_metrics.to_csv(table_path, index=False)
    logger.info("Saved attack metrics table to %s", table_path)

    # 2. Export metadata JSON
    meta_path = SAVED_MODELS_DIR / "attack_metadata_baseline.json"
    attack_metadata = {
        "attack_type": "Black-Box Shadow-Model Membership Inference Attack",
        "methodology": "Shokri et al. (2017) / Salem et al. (2019)",
        "shadow_models": {
            "count": attack_model.n_shadow_models,
            "architecture": "RandomForestClassifier",
            "train_ratio": attack_model.shadow_train_ratio,
        },
        "meta_classifier": {
            "model": "LogisticRegression(penalty='l2')",
            "features": FEATURE_NAMES_MIA,
            "coefficients": attack_model.meta_classifier.coef_[0].tolist(),
            "intercept": float(attack_model.meta_classifier.intercept_[0]),
        },
        "evaluation_cohort": {
            "n_members": eval_results["n_members"],
            "n_non_members": eval_results["n_non_members"],
            "total": eval_results["n_members"] + eval_results["n_non_members"],
        },
        "metrics": {
            "roc_auc": round(eval_results["roc_auc"], 4),
            "pr_auc": round(eval_results["pr_auc"], 4),
            "advantage": round(eval_results["advantage"], 4),
            "privacy_leakage": round(eval_results["privacy_leakage"], 4),
            "accuracy": round(eval_results["accuracy"], 4),
            "precision": round(eval_results["precision"], 4),
            "recall": round(eval_results["recall"], 4),
            "f1": round(eval_results["f1"], 4),
            "mean_member_score": round(eval_results["mean_member_score"], 4),
            "mean_non_member_score": round(eval_results["mean_non_member_score"], 4),
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(attack_metadata, f, indent=2)
    logger.info("Saved attack metadata JSON to %s", meta_path)

    # 3. Serialize fitted attack model
    attack_model_path = SAVED_MODELS_DIR / "shadow_mia_model.pkl"
    joblib.dump(attack_model, attack_model_path)
    logger.info("Serialized shadow MIA model to %s", attack_model_path)

    return {
        "metrics_table": str(table_path),
        "metadata_json": str(meta_path),
        "attack_model": str(attack_model_path),
    }


def run_attack_pipeline() -> Dict[str, Any]:
    """Execute end-to-end Phase 6 Shadow-Model Membership Inference Attack."""
    logger.info("Initiating Phase 6: Shadow-Model Membership Inference Attack...")

    # Step 1: Load preprocessed genomic matrices and baseline target model
    X_train, X_test, y_train, y_test, feature_names, target_model = (
        load_genomic_data_and_target_model()
    )

    # Step 2: Instantiate and fit shadow-model attack
    mia = ShadowModelMIA(
        n_shadow_models=10,
        shadow_train_ratio=0.5,
        meta_classifier_C=1.0,
        random_state=42,
    )
    mia.fit(X_train, y_train)

    # Step 3: Evaluate on real target model (Members = X_train, Non-Members = X_test)
    eval_results = mia.evaluate_on_target(
        target_model=target_model,
        X_members=X_train,
        y_members=y_train,
        X_non_members=X_test,
        y_non_members=y_test,
    )

    # Step 4: Generate publication-grade figures
    roc_fig_path = FIGURES_DIR / "attack_roc_curve.png"
    dist_fig_path = FIGURES_DIR / "attack_score_distribution.png"

    plot_attack_roc_curve(
        y_eval_true=eval_results["y_eval_true"],
        y_eval_scores=eval_results["y_eval_scores"],
        roc_auc=eval_results["roc_auc"],
        advantage=eval_results["advantage"],
        output_path=roc_fig_path,
    )

    plot_attack_score_distribution(
        scores_members=eval_results["scores_members"],
        scores_non_members=eval_results["scores_non_members"],
        output_path=dist_fig_path,
    )

    # Step 5: Export quantitative tables, metadata JSON, and serialized attack model
    artifacts = export_attack_artifacts(eval_results, mia)

    logger.info("Phase 6 Shadow-Model MIA successfully executed and exported.")
    return {
        "eval_results": eval_results,
        "roc_fig": str(roc_fig_path),
        "dist_fig": str(dist_fig_path),
        "artifacts": artifacts,
    }


if __name__ == "__main__":
    run_attack_pipeline()
