"""Plots for Member 3 baseline evaluation (displayed, never fabricated)."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

from src.evaluation.metrics import EvaluationResult


def plot_confusion(result: EvaluationResult) -> plt.Figure:
    """Show observed counts with negative/positive labels in that order."""
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(
        confusion_matrix=result.confusion,
        display_labels=[str(label) for label in result.labels],
    ).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{result.name}: confusion matrix")
    fig.tight_layout()
    return fig


def plot_roc_and_pr(results: Sequence[EvaluationResult]) -> plt.Figure:
    """Compare score curves using the real labels and predicted probabilities."""
    if not results:
        raise ValueError("Supply at least one evaluation result")
    fig, (roc_ax, pr_ax) = plt.subplots(1, 2, figsize=(11, 4))
    for result in results:
        binary_truth = np.asarray(result.y_true) == result.labels[1]
        if np.unique(binary_truth).size < 2:
            raise ValueError(f"{result.name}: ROC/PR plots require both classes")
        RocCurveDisplay.from_predictions(
            binary_truth, result.positive_scores, name=result.name, ax=roc_ax
        )
        PrecisionRecallDisplay.from_predictions(
            binary_truth, result.positive_scores, name=result.name, ax=pr_ax
        )
    roc_ax.set_title("ROC curve")
    pr_ax.set_title("Precision-recall curve")
    fig.tight_layout()
    return fig


def plot_metric_comparison(results: Sequence[EvaluationResult]) -> plt.Figure:
    """Compare held-out classification metrics at the same 0.5 threshold."""
    if not results:
        raise ValueError("Supply at least one evaluation result")
    names = ("accuracy", "precision", "recall", "f1", "roc_auc", "average_precision")
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 4))
    width = 0.8 / len(results)
    for i, result in enumerate(results):
        values = [np.nan if result.metrics[key] is None else result.metrics[key] for key in names]
        ax.bar(x - 0.4 + width * (i + 0.5), values, width, label=result.name)
    ax.set_xticks(x, ["Accuracy", "Precision", "Recall", "F1", "ROC AUC", "AP"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Same test split; positive-class metrics")
    ax.legend()
    fig.tight_layout()
    return fig
