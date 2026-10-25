"""De-duplicated benchmark construction (no canonical request in two splits).

Protocol:
  1. Unify sources into one request table.
  2. Compute a canonical key per request (identifiers/cookies removed, fixpoint-normalised).
  3. Drop keys that occur with both labels (label noise).
  4. De-duplicate: one row per canonical key across all sources.
  5. Keys seen at least 20 times in the raw traffic go to training (background rule; off with
     --strict-split); the rest is split 70/15/15 into train/val/test, stratified by
     (source, label, attack_type), with a seed.
  6. Assert zero canonical-key overlap between any two splits, including across sources.
  7. Write a manifest (counts, input file hashes, seed) next to the splits.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .normalize import add_text_views, canonical_key, twin_key


def file_md5(path: str | Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def add_canonical(df: pd.DataFrame, twin_merge: bool = False) -> pd.DataFrame:
    """twin_merge=True keys requests with normalize.twin_key (GET/POST twins become one request)."""
    df = df.copy()
    kf = twin_key if twin_merge else canonical_key
    df["key"] = [kf(m, p, q, b) for m, p, q, b in zip(df["method"], df["path"], df["query"], df["body"])]
    return add_text_views(df)


def dedup_and_clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    stats = {"rows_in": int(len(df))}
    conflict = df.groupby("key")["label"].nunique()
    bad = set(conflict[conflict > 1].index)
    stats["label_conflict_keys_dropped"] = int(len(bad))
    df = df[~df["key"].isin(bad)]
    before = len(df)
    df = df.copy()
    if "raw_count" in df.columns:
        df["raw_count"] = df["raw_count"].fillna(1).astype(int)
    else:
        df["raw_count"] = 1
    df["dup_count"] = df.groupby("key")["raw_count"].transform("sum")
    # cross-source duplicates (e.g. the same payload in HttpParams and Web Application Payloads)
    n_src = df.groupby("key")["source"].nunique()
    stats["cross_source_duplicate_keys"] = int((n_src > 1).sum())
    df = df.drop_duplicates(subset=["key"], keep="first")
    stats["duplicates_dropped"] = int(before - len(df))
    stats["rows_out"] = int(len(df))
    return df.reset_index(drop=True), stats


def stratified_split(df: pd.DataFrame, seed: int, frac_val: float = 0.15, frac_test: float = 0.15,
                     background_min_dup: int | None = 20) -> pd.DataFrame:
    """Stratified split over unique canonical requests.

    background_min_dup: canonical requests that occurred >= this many times in the raw traffic
    are 'application background' (parameterless pages, fixed form submissions, templated
    attacks).  A deployed detector has necessarily seen them in its training history, so they
    are assigned to the training split.  Every key still lives in exactly one split, so this
    does not create leakage; it only removes the artefact that a popular request looks novel.
    Set to None for the strict protocol (pure random split of unique requests).
    """
    df = df.copy()
    levels = [df["source"] + "|" + df["label"].astype(str) + "|" + df["attack_type"].astype(str),
              df["source"] + "|" + df["label"].astype(str),
              df["label"].astype(str)]

    def strata(idx, min_count=6):
        """Finest stratum with >= min_count members within idx (hierarchical collapse)."""
        s = levels[0].iloc[idx].copy()
        for lvl in levels[1:]:
            vc = s.value_counts()
            rare = s.isin(set(vc[vc < min_count].index))
            if not rare.any():
                break
            s[rare] = lvl.iloc[idx][rare]
        vc = s.value_counts()
        rare = s.isin(set(vc[vc < min_count].index))
        if rare.any():  # last resort: merge leftovers into the largest stratum
            s[rare] = vc.index[0]
        return s.to_numpy()

    split = np.array([""] * len(df), dtype=object)
    bg = np.zeros(len(df), dtype=bool)
    if background_min_dup is not None and "dup_count" in df.columns:
        bg = (df["dup_count"].to_numpy() >= background_min_dup)
    split[bg] = "train"
    idx = np.where(~bg)[0]
    tr, rest = train_test_split(idx, test_size=frac_val + frac_test, random_state=seed, stratify=strata(idx))
    va, te = train_test_split(rest, test_size=frac_test / (frac_val + frac_test), random_state=seed,
                              stratify=strata(rest, min_count=2))
    split[tr] = "train"
    split[va] = "val"
    split[te] = "test"
    df["split"] = split
    df["background"] = bg
    return df


def assert_no_overlap(df: pd.DataFrame) -> dict:
    out = {}
    keys = {s: set(df.loc[df["split"] == s, "key"]) for s in ("train", "val", "test")}
    for a in keys:
        for b in keys:
            if a < b:
                n = len(keys[a] & keys[b])
                out[f"{a}&{b}"] = n
                assert n == 0, f"canonical-key overlap between {a} and {b}: {n}"
    return out


def build(sources: dict[str, pd.DataFrame], seed: int, out_dir: str | Path, input_files: dict[str, str] | None = None,
          background_min_dup: int | None = 20, twin_merge: bool = False) -> pd.DataFrame:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.concat(sources.values(), ignore_index=True)
    df = add_canonical(df, twin_merge=twin_merge)
    df, stats = dedup_and_clean(df)
    stats["twin_merge"] = bool(twin_merge)
    df = stratified_split(df, seed=seed, background_min_dup=background_min_dup)
    overlaps = assert_no_overlap(df)
    stats["background_min_dup"] = background_min_dup
    stats["background_keys_forced_to_train"] = {str(k): int(v) for k, v in
                                                 df[df["background"]].groupby("label").size().items()}
    manifest = {
        "seed": seed,
        "stats": stats,
        "overlap_checks": overlaps,
        "counts": {k: v for k, v in df.groupby(["source", "split", "label"]).size().reset_index(name="n")
                   .apply(lambda r: (f"{r['source']}|{r['split']}|{r['label']}", int(r['n'])), axis=1)},
        "inputs": {k: file_md5(v) for k, v in (input_files or {}).items()},
    }
    (out_dir / f"manifest_seed{seed}.json").write_text(json.dumps(manifest, indent=2))
    df.to_pickle(out_dir / f"bench_seed{seed}.pkl")
    return df
