"""Reproducible text-only control for the CrossHealth-Risk experiments.

The input column names belong to the standardized dataset interface supplied
by Member 1. Neither this module nor its notebooks assume FakeHealth's schema.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class DataSplit:
    """One fixed, stratified split reused by all baseline experiments."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def split_labeled_data(
    frame: pd.DataFrame,
    *,
    text_column: str,
    label_column: str,
    test_size: float = 0.20,
    validation_size: float = 0.20,
    random_state: int = 42,
) -> DataSplit:
    """Make disjoint train/validation/test splits (60/20/20 by default).

    Both splits are stratified. The caller must remove duplicates and decide
    which real columns constitute the text and binary ground-truth label.
    Splitting happens before any fitted text transformation or class weighting.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if not text_column or not label_column or text_column == label_column:
        raise ValueError("Supply distinct, nonempty text_column and label_column")
    missing = [name for name in (text_column, label_column) if name not in frame]
    if missing:
        raise KeyError(f"Missing column(s) {missing}; available: {list(frame.columns)}")
    if not 0 < test_size < 1 or not 0 < validation_size < 1 - test_size:
        raise ValueError("test_size and validation_size must be positive and sum to less than 1")
    if frame[[text_column, label_column]].isna().any().any():
        raise ValueError("Text and labels contain missing values; finish Member 1 preprocessing first")
    if not frame[text_column].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError("Text must be nonempty strings; finish Member 1 preprocessing first")
    if frame[label_column].nunique() != 2:
        raise ValueError("The binary baseline needs exactly two observed label values")

    try:
        development, test = train_test_split(
            frame, test_size=test_size, stratify=frame[label_column], random_state=random_state
        )
        train, validation = train_test_split(
            development,
            test_size=validation_size / (1 - test_size),
            stratify=development[label_column],
            random_state=random_state,
        )
    except ValueError as exc:
        raise ValueError("Cannot stratify this dataset; check class counts and split sizes") from exc

    for name, part in (("training", train), ("validation", validation), ("test", test)):
        if part[label_column].nunique() != 2:
            raise ValueError(f"The {name} split lacks a class; use more examples or larger splits")
    return DataSplit(*(part.reset_index(drop=True) for part in (train, validation, test)))


def build_baseline(
    *,
    max_features: int = 5000,
    c: float = 1.0,
    class_weight: str | None = None,
    random_state: int = 42,
) -> Pipeline:
    """Create a TF-IDF + logistic regression classifier.

    The unweighted baseline uses ``class_weight=None``. Fit the *whole*
    pipeline on training text only; validation and test call ``predict`` or
    ``predict_proba`` without ever refitting the vectorizer.
    """
    if max_features < 1 or c <= 0:
        raise ValueError("max_features and c must be positive")
    if class_weight not in (None, "balanced"):
        raise ValueError("class_weight must be None or 'balanced'")
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    max_features=max_features,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    solver="liblinear",
                    max_iter=2000,
                    C=c,
                    class_weight=class_weight,
                    random_state=random_state,
                ),
            ),
        ]
    )
