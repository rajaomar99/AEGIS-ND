"""AEGIS-ND Adversarial Attack Package."""

from src.attacks.membership_inference import (
    ShadowModelMIA,
    extract_attack_features,
    run_attack_pipeline,
)
from src.attacks.attribute_inference import (
    GenomicAttributeInferenceAttack,
    run_attribute_inference_pipeline,
)

__all__ = [
    "ShadowModelMIA",
    "extract_attack_features",
    "run_attack_pipeline",
    "GenomicAttributeInferenceAttack",
    "run_attribute_inference_pipeline",
]

