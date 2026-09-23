"""Class-weighted counterpart to the text-only control model."""

from __future__ import annotations

from sklearn.pipeline import Pipeline

from src.models.baseline import build_baseline


def build_balanced_baseline(
    *, max_features: int = 5000, c: float = 1.0, random_state: int = 42
) -> Pipeline:
    """Use inverse-frequency class weights learned only from training labels.

    All other features, hyperparameters, and split rules match the unweighted
    baseline so the comparison isolates imbalance handling. No validation or
    test examples are resampled or used to calculate the class weights.
    """
    return build_baseline(
        max_features=max_features,
        c=c,
        class_weight="balanced",
        random_state=random_state,
    )
