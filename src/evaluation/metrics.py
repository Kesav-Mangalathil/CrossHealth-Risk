"""Binary classifier evaluation with an explicitly chosen positive class."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class EvaluationResult:
    name: str
    metrics: dict[str, float | int | None]
    confusion: np.ndarray  # [[TN, FP], [FN, TP]]
    labels: tuple[Any, Any]  # (negative, positive)
    y_true: np.ndarray
    y_pred: np.ndarray
    positive_scores: np.ndarray


def evaluate_binary_classifier(
    model: Any, texts: Any, y_true: Any, *, positive_label: Any, name: str
) -> EvaluationResult:
    """Evaluate a fitted estimator on an untouched validation or test split.

    ROC AUC and average precision require probabilities for the named positive
    class. They return None when the evaluation split contains only one class.
    Average precision is a useful complement to ROC AUC under imbalance.
    """
    classes = np.asarray(model.classes_)
    if len(classes) != 2 or positive_label not in classes:
        raise ValueError(f"positive_label must be one of the two fitted classes: {classes.tolist()}")
    negative_label = next(label for label in classes if label != positive_label)
    truth = np.asarray(y_true)
    if truth.ndim != 1 or len(truth) == 0 or len(truth) != len(texts):
        raise ValueError("y_true and texts must be nonempty and have matching lengths")
    if not np.isin(truth, classes).all():
        raise ValueError("Evaluation labels must match the training classes")

    predicted = np.asarray(model.predict(texts))
    probabilities = np.asarray(model.predict_proba(texts))
    if probabilities.shape != (len(truth), 2):
        raise ValueError("The classifier must return two probabilities per example")
    scores = probabilities[:, int(np.where(classes == positive_label)[0][0])]
    binary_truth = truth == positive_label

    metrics: dict[str, float | int | None] = {
        "n": int(len(truth)),
        "positive_count": int(binary_truth.sum()),
        "accuracy": float(accuracy_score(truth, predicted)),
        "precision": float(precision_score(truth, predicted, pos_label=positive_label, zero_division=0)),
        "recall": float(recall_score(truth, predicted, pos_label=positive_label, zero_division=0)),
        "f1": float(f1_score(truth, predicted, pos_label=positive_label, zero_division=0)),
        "roc_auc": None,
        "average_precision": None,
    }
    if np.unique(binary_truth).size == 2:
        metrics["roc_auc"] = float(roc_auc_score(binary_truth, scores))
        metrics["average_precision"] = float(average_precision_score(binary_truth, scores))
    return EvaluationResult(
        name=name,
        metrics=metrics,
        confusion=confusion_matrix(truth, predicted, labels=[negative_label, positive_label]),
        labels=(negative_label, positive_label),
        y_true=truth,
        y_pred=predicted,
        positive_scores=scores,
    )
