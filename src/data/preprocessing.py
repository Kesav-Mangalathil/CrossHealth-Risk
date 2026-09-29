"""
Preprocessing for CrossHealth-Risk (Member 1).

Output contract for Members 2-4 (files in data/processed/<dataset>_processed.csv):
    id, dataset, subset, title, text, label, label_raw, source,
    n_tweets, n_replies, n_retweets, has_source, has_engagement, has_network,
    split, text_len, word_count
`text` is cleaned but NOT lower-cased (capitalisation is a feature for Member 2).
Missing engagement counts stay NaN (NOT 0) so the context mask can tell
"unavailable" from "zero engagement".
"""
from __future__ import annotations

import html
import re
import unicodedata
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

_URL = re.compile(r"https?://\S+|www\.\S+")
_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")
_ZW = re.compile(r"[\u200b\u200c\u200d\ufeff]")


def clean_text(s) -> str:
    if s is None or s is pd.NA or (isinstance(s, float) and pd.isna(s)):
        return ""
    s = html.unescape(str(s))
    s = unicodedata.normalize("NFKC", s)
    s = _ZW.sub("", s)
    s = _TAG.sub(" ", s)
    s = _URL.sub(" ", s)
    return _WS.sub(" ", s).strip()


def handle_missing(df: pd.DataFrame, min_words: int = 5) -> pd.DataFrame:
    """Clean text/title, fall back to title when body is empty, drop unlabeled or too-short rows."""
    df = df.copy()
    df["title"] = df["title"].map(clean_text)
    df["text"] = df["text"].map(clean_text)
    empty = df["text"].str.len() == 0
    df.loc[empty, "text"] = df.loc[empty, "title"]
    df = df[df["label"].notna()].copy()
    df["label"] = df["label"].astype(int)
    df = df[df["text"].str.split().str.len() >= min_words]
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop repeated ids and repeated texts (case-insensitive). Keeps the first occurrence."""
    df = df.drop_duplicates(subset="id")
    return df.loc[~df["text"].str.lower().duplicated()]


def add_text_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text_len"] = df["text"].str.len()
    df["word_count"] = df["text"].str.split().str.len()
    return df


def _can_stratify(y) -> bool:
    vc = y.value_counts()
    return len(vc) > 1 and vc.min() >= 3


def split_dataset(df: pd.DataFrame, val_size=0.15, test_size=0.15, seed=42) -> pd.DataFrame:
    """Stratified train/val/test split by label."""
    df = df.copy()
    train_val, test = train_test_split(
        df, test_size=test_size, random_state=seed,
        stratify=df["label"] if _can_stratify(df["label"]) else None)
    train, val = train_test_split(
        train_val, test_size=val_size / (1 - test_size), random_state=seed,
        stratify=train_val["label"] if _can_stratify(train_val["label"]) else None)
    df["split"] = ""
    df.loc[train.index, "split"] = "train"
    df.loc[val.index, "split"] = "val"
    df.loc[test.index, "split"] = "test"
    return df


def preprocess_dataset(df: pd.DataFrame, min_words: int = 5, seed: int = 42, verbose=True) -> pd.DataFrame:
    n0 = len(df)
    df = handle_missing(df, min_words=min_words); n1 = len(df)
    df = remove_duplicates(df); n2 = len(df)
    df = add_text_stats(df)
    df = split_dataset(df, seed=seed)
    if verbose:
        print(f"rows: {n0} -> {n1} after missing/short filter -> {n2} after de-duplication")
        print(df.groupby("split")["label"].value_counts().unstack(fill_value=0))
    return df.reset_index(drop=True)


def save_processed(df: pd.DataFrame, name: str, out_dir="data/processed") -> Path:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}_processed.csv"
    df.to_csv(path, index=False)
    return path


def load_processed(name: str, in_dir="data/processed") -> pd.DataFrame:
    """Entry point for Members 2-4."""
    return pd.read_csv(Path(in_dir) / f"{name}_processed.csv")
