"""AEGIS-ND Evaluation & Synthesis Package."""

from src.evaluation.metrics import (
    compute_diagnostic_metrics,
    compute_privacy_leakage_metrics,
    compute_attribute_inversion_metrics,
)
from src.evaluation.synthesis import (
    build_master_evaluation_matrix,
    run_synthesis_pipeline,
)
from src.evaluation.visualize import plot_radar_benchmark

__all__ = [
    "compute_diagnostic_metrics",
    "compute_privacy_leakage_metrics",
    "compute_attribute_inversion_metrics",
    "build_master_evaluation_matrix",
    "run_synthesis_pipeline",
    "plot_radar_benchmark",
]

