"""
Dataset loading for CrossHealth-Risk (Member 1).

Every loader returns a DataFrame in the STANDARD SCHEMA below, so that
Members 2-4 never need to know about the raw format of each dataset.

Label convention (all datasets):  0 = reliable / real,  1 = misinformation / fake

NOTE: The raw-format assumptions (folder names, column names) follow the public
releases of each dataset. Run `inspect_dataframe` / the notebook 01 checks on your
local copy and adjust the constants below if your copy differs.
"""
from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

# ---------------------------------------------------------------- schema
STANDARD_COLUMNS = [
    "id",               # unique id within dataset
    "dataset",          # fakehealth | coaid | pubhealth
    "subset",           # e.g. HealthStory / HealthRelease / news / claim
    "title",            # headline or claim (may be empty)
    "text",             # main body text
    "label",            # 0 real, 1 fake
    "label_raw",        # original label / rating, for traceability
    "source",           # publisher / domain, NaN if unavailable
    "n_tweets",         # engagement counts, NaN if unavailable
    "n_replies",
    "n_retweets",
    "has_source",       # availability flags (used by Member 2 for the context mask)
    "has_engagement",
    "has_network",
    "split",            # filled by preprocessing (train / val / test)
]

# PUBHEALTH label handling
PUBHEALTH_LABEL_MAP = {"false": 1, "true": 0}       # 'mixture' / 'unproven' -> dropped by default
# FakeHealth: HealthNewsReview star rating (0-5). rating < threshold => fake
FAKEHEALTH_FAKE_BELOW = 3


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    for c in STANDARD_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    for c in ("n_tweets", "n_replies", "n_retweets"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["has_source"] = df["source"].notna() & (df["source"].astype(str).str.strip() != "")
    df["has_engagement"] = df[["n_tweets", "n_replies", "n_retweets"]].notna().any(axis=1)
    df["has_network"] = df["has_network"].fillna(False).astype(bool)
    return df[STANDARD_COLUMNS].reset_index(drop=True)


# ---------------------------------------------------------------- FakeHealth
def load_fakehealth(root, subsets=("HealthStory", "HealthRelease"),
                    fake_below: float = FAKEHEALTH_FAKE_BELOW) -> pd.DataFrame:
    """
    Expected layout (Dai et al., 2020):
        root/content/<subset>/<news_id>.json      (keys: title, text, ...)
        root/reviews/<subset>.json                (list of dicts: news_id, rating, ...)
        root/engagements/<subset>/<news_id>.json  (keys: tweets, replies, retweets)  [optional]
        root/user_network/                        [optional -> has_network]
    """
    root = Path(root)
    if (root / "dataset").exists():
        root = root / "dataset"

    eng_root = next((root / n for n in ("engagements", "engagement") if (root / n).exists()), None)
    has_network = (root / "user_network").exists()

    frames = []
    for sub in subsets:
        review_file = root / "reviews" / f"{sub}.json"
        content_dir = root / "content" / sub
        if not review_file.exists() or not content_dir.exists():
            print(f"[fakehealth] skipping {sub}: missing {review_file} or {content_dir}")
            continue
        reviews = json.loads(review_file.read_text(encoding="utf-8"))
        rows = []
        for rev in reviews:
            nid = str(rev.get("news_id"))
            cfile = content_dir / f"{nid}.json"
            if not cfile.exists():
                continue
            content = json.loads(cfile.read_text(encoding="utf-8"))
            rating = rev.get("rating")
            row = dict(
                id=f"fakehealth_{sub}_{nid}", dataset="fakehealth", subset=sub,
                title=content.get("title"), text=content.get("text"),
                label=None if rating is None else int(float(rating) < fake_below),
                label_raw=rating,
                source=rev.get("news_source") or rev.get("source"),
                has_network=has_network,
            )
            if eng_root is not None:
                efile = eng_root / sub / f"{nid}.json"
                if efile.exists():
                    e = json.loads(efile.read_text(encoding="utf-8"))
                    row.update(n_tweets=len(e.get("tweets", [])),
                               n_replies=len(e.get("replies", [])),
                               n_retweets=len(e.get("retweets", [])))
            rows.append(row)
        frames.append(pd.DataFrame(rows))
    if not frames:
        raise FileNotFoundError(f"No FakeHealth data found under {root}")
    return _finalize(pd.concat(frames, ignore_index=True))


# ---------------------------------------------------------------- CoAID
def load_coaid(root, include_claims: bool = False) -> pd.DataFrame:
    """
    Expected layout: root/<snapshot-date>/{News,Claim}{Fake,Real}COVID-19.csv
    News files columns: id, news_url, title, content, abstract, ..., tweet_id (tab-separated)
    CoAID snapshots are cumulative, so duplicates across folders are expected and
    are removed in preprocessing.
    """
    root = Path(root)
    kinds = ["News"] + (["Claim"] if include_claims else [])
    rows = []
    for kind in kinds:
        for lab_name, lab in (("Fake", 1), ("Real", 0)):
            for f in sorted(root.rglob(f"{kind}{lab_name}COVID-19.csv")):
                df = pd.read_csv(f)
                snap = f.parent.name
                for i, r in df.iterrows():
                    tweets = r.get("tweet_id")
                    n_tw = len(str(tweets).split("\t")) if pd.notna(tweets) else pd.NA
                    url = r.get("news_url")
                    src = urlparse(str(url)).netloc.replace("www.", "") if pd.notna(url) else pd.NA
                    body = r.get("content")
                    if pd.isna(body):
                        body = r.get("abstract")
                    rows.append(dict(
                        id=f"coaid_{snap}_{kind}_{lab_name}_{r.get('id', i)}",
                        dataset="coaid", subset=kind.lower(),
                        title=r.get("title"), text=body,
                        label=lab, label_raw=f"{kind}{lab_name}",
                        source=src, n_tweets=n_tw,
                    ))
    if not rows:
        raise FileNotFoundError(f"No CoAID csv files found under {root}")
    return _finalize(pd.DataFrame(rows))


# ---------------------------------------------------------------- PUBHEALTH
def load_pubhealth(root, keep_labels=("true", "false")) -> pd.DataFrame:
    """
    Expected: root/{train,dev,test}.tsv  with columns
    claim_id, claim, date_published, explanation, fact_checkers, main_text, sources, label, subjects
    Only 'true'/'false' are mapped to binary by default; 'mixture'/'unproven' are dropped.
    PUBHEALTH has no engagement/network information.
    """
    root = Path(root)
    frames = []
    for name in ("train", "dev", "test"):
        f = root / f"{name}.tsv"
        if f.exists():
            frames.append(pd.read_csv(f, sep="\t"))
    if not frames:
        raise FileNotFoundError(f"No PUBHEALTH tsv files found under {root}")
    raw = pd.concat(frames, ignore_index=True)
    raw = raw[raw["label"].isin(keep_labels)]
    out = pd.DataFrame(dict(
        id=["pubhealth_" + str(x) for x in raw["claim_id"]],
        dataset="pubhealth", subset="claim",
        title=raw["claim"], text=raw["main_text"],
        label=raw["label"].map(PUBHEALTH_LABEL_MAP), label_raw=raw["label"],
    ))
    return _finalize(out)


LOADERS = {"fakehealth": load_fakehealth, "coaid": load_coaid, "pubhealth": load_pubhealth}


def load_dataset(name: str, root, **kw) -> pd.DataFrame:
    return LOADERS[name.lower()](root, **kw)


# ---------------------------------------------------------------- inspection
def inspect_dataframe(df: pd.DataFrame, name: str = "") -> dict:
    """Rows, columns, dtypes, missing values, duplicates, label distribution."""
    text = df["text"].fillna("").astype(str)
    return {
        "name": name,
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_frame("dtype"),
        "missing": pd.DataFrame({"missing": df.isna().sum(),
                                 "pct": (df.isna().mean() * 100).round(2)}),
        "duplicate_ids": int(df["id"].duplicated().sum()),
        "duplicate_text": int(text[text.str.strip() != ""].duplicated().sum()),
        "empty_text": int((text.str.strip() == "").sum()),
        "label_counts": df["label"].value_counts(dropna=False).sort_index(),
        "label_pct": (df["label"].value_counts(normalize=True, dropna=False) * 100).round(2).sort_index(),
        "context_available": df[["has_source", "has_engagement", "has_network"]].mean().round(3),
    }


def print_inspection(summary: dict) -> None:
    print(f"===== {summary['name']} =====")
    print("shape:", summary["shape"])
    for k in ("dtypes", "missing", "label_counts", "label_pct", "context_available"):
        print(f"\n[{k}]\n{summary[k]}")
    print(f"\nduplicate ids: {summary['duplicate_ids']} | duplicate texts: "
          f"{summary['duplicate_text']} | empty texts: {summary['empty_text']}")
