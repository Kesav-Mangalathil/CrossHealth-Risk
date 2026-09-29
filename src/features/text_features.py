"""
Reusable text feature engineering for CrossHealth-Risk.

This module intentionally does not assume a particular dataset schema. The caller
must provide the name of the standardized text column created by Member 1.

Outputs:
- TF-IDF features for downstream ML models.
- Lightweight text statistics that can be used as additional numeric features.

The module does not load raw datasets and does not invent dataset columns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class TextFeatureResult:
    """Container for text features and the fitted TF-IDF vectorizer."""
    tfidf_matrix: object
    statistics: pd.DataFrame
    feature_names: list[str]
    vectorizer: TfidfVectorizer


def _validate_text_column(df: pd.DataFrame, text_column: str) -> None:
    if text_column not in df.columns:
        available = ", ".join(map(str, df.columns))
        raise KeyError(
            f"Text column '{text_column}' was not found. "
            f"Available columns: {available}"
        )


def _safe_text_series(df: pd.DataFrame, text_column: str) -> pd.Series:
    _validate_text_column(df, text_column)
    return df[text_column].fillna("").astype(str)


def extract_text_statistics(texts: pd.Series) -> pd.DataFrame:
    """Create deterministic, model-agnostic statistics from a text series."""
    texts = texts.fillna("").astype(str)

    words = texts.str.findall(r"\b\w+\b")
    word_count = words.str.len().astype(float)
    char_count = texts.str.len().astype(float)

    # Avoid division by zero for empty texts.
    avg_word_length = (
        texts.str.findall(r"\b\w+\b")
        .apply(lambda tokens: float(np.mean([len(t) for t in tokens])) if tokens else 0.0)
    )

    alpha_chars = texts.str.count(r"[A-Za-z]").astype(float)
    uppercase_chars = texts.apply(lambda value: sum(ch.isupper() for ch in value)).astype(float)
    digit_chars = texts.str.count(r"\d").astype(float)

    statistics = pd.DataFrame(
        {
            "text_char_count": char_count,
            "text_word_count": word_count,
            "text_sentence_count": texts.str.count(r"[.!?]+").astype(float),
            "text_avg_word_length": avg_word_length,
            "text_uppercase_ratio": np.where(alpha_chars > 0, uppercase_chars / alpha_chars, 0.0),
            "text_digit_ratio": np.where(char_count > 0, digit_chars / char_count, 0.0),
            "text_url_count": texts.str.count(r"https?://\S+|www\.\S+").astype(float),
            "text_question_count": texts.str.count(r"\?").astype(float),
            "text_exclamation_count": texts.str.count(r"!").astype(float),
            "text_empty_flag": (word_count == 0).astype(float),
        },
        index=texts.index,
    )

    return statistics


def fit_text_features(
    df: pd.DataFrame,
    text_column: str,
    *,
    max_features: int = 5000,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int | float = 1,
    sublinear_tf: bool = True,
) -> TextFeatureResult:
    """
    Fit TF-IDF on a standardized dataset and return text features.

    The vectorizer must be fitted only on the training split in a real experiment
    to avoid train/test leakage. Use ``transform_text_features`` for validation,
    test, or cross-dataset data after fitting on training data.
    """
    texts = _safe_text_series(df, text_column)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=sublinear_tf,
    )

    matrix = vectorizer.fit_transform(texts)
    statistics = extract_text_statistics(texts)
    feature_names = vectorizer.get_feature_names_out().tolist()

    return TextFeatureResult(
        tfidf_matrix=matrix,
        statistics=statistics,
        feature_names=feature_names,
        vectorizer=vectorizer,
    )


def transform_text_features(
    df: pd.DataFrame,
    text_column: str,
    vectorizer: TfidfVectorizer,
) -> tuple[object, pd.DataFrame]:
    """Transform new data using an already-fitted TF-IDF vectorizer."""
    texts = _safe_text_series(df, text_column)
    matrix = vectorizer.transform(texts)
    statistics = extract_text_statistics(texts)
    return matrix, statistics
