"""Split train/val/test POR VIDEO (nunca se mezclan frames del mismo video)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def split_videos(
    videos: list[str], val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 0
) -> dict[str, str]:
    """Asigna cada video a un split.

    - 1 video: todo train (aviso). 2 videos: train + val. >=3: train/val/test con >=1 video cada uno.
    """
    vids = sorted(set(videos))
    n = len(vids)
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(n))
    if n == 1:
        warnings.warn("sólo hay 1 video: no hay val/test; todo queda en train", stacklevel=2)
        return {vids[0]: "train"}
    if n == 2:
        n_val, n_test = 1, 0
    else:
        n_val = max(1, round(n * val_frac))
        n_test = max(1, round(n * test_frac))
        while n - n_val - n_test < 1:
            n_val, n_test = max(1, n_val - 1), max(1, n_test - 1)
    out: dict[str, str] = {}
    for rank, i in enumerate(order):
        out[vids[i]] = "val" if rank < n_val else "test" if rank < n_val + n_test else "train"
    return out


def assign_splits(
    df: pd.DataFrame, val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 0
) -> pd.DataFrame:
    df = df.copy()
    if df.empty:
        return df
    mapping = split_videos(list(df["video"].unique()), val_frac, test_frac, seed)
    df["split"] = df["video"].map(mapping)
    check_no_leakage(df)
    return df


def check_no_leakage(df: pd.DataFrame) -> None:
    per_video = df.groupby("video")["split"].nunique()
    bad = per_video[per_video > 1]
    if len(bad):
        raise ValueError(f"videos en más de un split: {list(bad.index)}")
