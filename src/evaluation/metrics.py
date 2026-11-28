"""AEGIS-ND: Evaluation Metrics & Privacy Loss Quantification Functions
=============================================================================
Defines standard utility, adversarial advantage, and information-theoretic
privacy metrics across the AEGIS-ND research framework.

Author: Raja Omar
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_diagnostic_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Compute comprehensive clinical diagnostic classification metrics."""
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, y_proba)),
    }


def compute_privacy_leakage_metrics(y_mia_true: np.ndarray, mia_scores: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Compute cryptographic advantage, empirical leakage, and membership inference metrics."""
    y_pred = (mia_scores >= threshold).astype(int)
    roc_auc = float(roc_auc_score(y_mia_true, mia_scores))
    return {
        "attack_roc_auc": roc_auc,
        "attack_pr_auc": float(average_precision_score(y_mia_true, mia_scores)),
        "adversarial_advantage": float(2.0 * (roc_auc - 0.5)),
        "empirical_privacy_leakage": float(roc_auc - 0.5),
        "attack_accuracy": float(accuracy_score(y_mia_true, y_pred)),
        "attack_precision": float(precision_score(y_mia_true, y_pred, zero_division=0)),
        "attack_recall": float(recall_score(y_mia_true, y_pred, zero_division=0)),
        "attack_f1": float(f1_score(y_mia_true, y_pred, zero_division=0)),
    }


def compute_attribute_inversion_metrics(s_true: np.ndarray, inversion_scores: np.ndarray, threshold: float = 0.0) -> dict[str, float]:
    """Compute attribute reconstruction metrics and information gain over majority prior."""
    s_pred = (inversion_scores >= threshold).astype(int)
    acc = float(accuracy_score(s_true, s_pred))
    base_rate = float(max(np.mean(s_true == 1), np.mean(s_true == 0)))
    return {
        "reconstruction_roc_auc": float(roc_auc_score(s_true, inversion_scores)),
        "reconstruction_pr_auc": float(average_precision_score(s_true, inversion_scores)),
        "reconstruction_accuracy": acc,
        "balanced_accuracy": float(balanced_accuracy_score(s_true, s_pred)),
        "reconstruction_precision": float(precision_score(s_true, s_pred, zero_division=0)),
        "reconstruction_recall": float(recall_score(s_true, s_pred, zero_division=0)),
        "reconstruction_f1": float(f1_score(s_true, s_pred, zero_division=0)),
        "base_rate_prior": base_rate,
        "adversarial_information_gain": float(acc - base_rate),
    }

