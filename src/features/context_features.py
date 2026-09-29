"""
Reusable contextual feature engineering for CrossHealth-Risk.

The module deliberately does not assume that a particular engagement, source,
or network column exists. The caller supplies the standardized column lists
created by Member 1.

A row-level context-availability mask is returned:
    1 -> at least one requested context field is present
    0 -> all requested context fields are missing.

For numerical context features, median imputation and standardization are
performed by a fitted sklearn pipeline. Categorical features are imputed and
one-hot encoded with unknown categories ignored during transform.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class ContextFeatureResult:
    """Container for encoded context features and metadata."""
    matrix: object
    feature_names: list[str]
    availability_mask: pd.Series
    availability_fraction: pd.Series
    preprocessor: ColumnTransformer


def _validate_columns(
    df: pd.DataFrame,
    columns: Sequence[str],
    group_name: str,
) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        available = ", ".join(map(str, df.columns))
        raise KeyError(
            f"Missing {group_name} column(s): {missing}. "
            f"Available columns: {available}"
        )


def create_context_availability_mask(
    df: pd.DataFrame,
    context_columns: Sequence[str],
) -> tuple[pd.Series, pd.Series]:
    """
    Return row-level context availability and the fraction of context fields
    that are available.

    The function only uses columns explicitly supplied by the caller.
    """
    context_columns = list(context_columns)
    if not context_columns:
        mask = pd.Series(0, index=df.index, dtype="int8", name="context_available")
        fraction = pd.Series(
            0.0, index=df.index, dtype="float32", name="context_availability_fraction"
        )
        return mask, fraction

    _validate_columns(df, context_columns, "context")
    present = df[context_columns].notna()

    # Treat empty strings in object columns as missing contextual information.
    for column in context_columns:
        if pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_string_dtype(df[column]):
            present[column] &= df[column].fillna("").astype(str).str.strip().ne("")

    availability_fraction = present.mean(axis=1).astype("float32")
    availability_mask = (availability_fraction > 0).astype("int8")
    availability_mask.name = "context_available"
    availability_fraction.name = "context_availability_fraction"

    return availability_mask, availability_fraction


def _make_one_hot_encoder() -> OneHotEncoder:
    """Create a version-compatible OneHotEncoder."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:  # Older scikit-learn
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def fit_context_features(
    df: pd.DataFrame,
    *,
    numeric_columns: Sequence[str] = (),
    categorical_columns: Sequence[str] = (),
) -> ContextFeatureResult:
    """
    Fit and transform the explicitly supplied contextual columns.

    ``numeric_columns`` and ``categorical_columns`` must come from the
    standardized Member 1 dataset interface. No columns are inferred.
    """
    numeric_columns = list(numeric_columns)
    categorical_columns = list(categorical_columns)
    context_columns = numeric_columns + categorical_columns

    if len(context_columns) != len(set(context_columns)):
        raise ValueError("A context column appears in more than one column list.")

    _validate_columns(df, context_columns, "context")

    if not context_columns:
        # Return an empty dense matrix while still providing the required mask.
        mask, fraction = create_context_availability_mask(df, context_columns)
        return ContextFeatureResult(
            matrix=np.empty((len(df), 0), dtype=np.float32),
            feature_names=[],
            availability_mask=mask,
            availability_fraction=fraction,
            preprocessor=None,  # type: ignore[arg-type]
        )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _make_one_hot_encoder()),
        ]
    )

    transformers = []
    if numeric_columns:
        transformers.append(("numeric", numeric_pipeline, numeric_columns))
    if categorical_columns:
        transformers.append(("categorical", categorical_pipeline, categorical_columns))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    matrix = preprocessor.fit_transform(df)
    feature_names = preprocessor.get_feature_names_out().tolist()
    mask, fraction = create_context_availability_mask(df, context_columns)

    return ContextFeatureResult(
        matrix=matrix,
        feature_names=feature_names,
        availability_mask=mask,
        availability_fraction=fraction,
        preprocessor=preprocessor,
    )


def transform_context_features(
    df: pd.DataFrame,
    *,
    numeric_columns: Sequence[str] = (),
    categorical_columns: Sequence[str] = (),
    preprocessor: ColumnTransformer,
) -> tuple[object, pd.Series, pd.Series]:
    """Transform new data using an already-fitted context preprocessor."""
    numeric_columns = list(numeric_columns)
    categorical_columns = list(categorical_columns)
    context_columns = numeric_columns + categorical_columns

    _validate_columns(df, context_columns, "context")

    matrix = preprocessor.transform(df)
    mask, fraction = create_context_availability_mask(df, context_columns)
    return matrix, mask, fraction
