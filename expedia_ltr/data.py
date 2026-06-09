from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from .config import (
    BOOKING_COL,
    CLICK_COL,
    GAIN_COL,
    GROUP_COL,
    LOGGER,
    TARGET_COL,
    TRAIN_ONLY_COLS,
)


def load_dataset(path: Path, sample_groups: int = 0, seed: int = 2026) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    LOGGER.info("Reading %s", path)
    df = pd.read_parquet(path)
    LOGGER.info("Loaded %s with shape %s", path.name, df.shape)
    if sample_groups > 0:
        df = sample_by_groups(df, sample_groups, seed)
        LOGGER.info(
            "Sampled %s groups from %s; shape is now %s",
            sample_groups,
            path.name,
            df.shape,
        )
    return df


def sample_by_groups(df: pd.DataFrame, sample_groups: int, seed: int) -> pd.DataFrame:
    """
    Randomly sample a subset of groups from the DataFrame, keeping all rows for the selected groups.
    """
    groups = pd.Series(df[GROUP_COL].unique())
    if sample_groups >= len(groups):
        return df
    keep = set(groups.sample(sample_groups, random_state=seed).to_numpy())
    return df[df[GROUP_COL].isin(keep)].copy()


def add_relevance(df: pd.DataFrame) -> None:
    """
    Add a relevance column based on booking and click indicators.
    Booking: 5, Click-only: 1, No interaction: 0
    """

    if BOOKING_COL not in df.columns or CLICK_COL not in df.columns:
        raise ValueError("Training data must include click_bool and booking_bool.")

    conditions: List[pd.Series[bool]] = [
        df[BOOKING_COL].eq(1),
        df[CLICK_COL].eq(1),
    ]

    df[TARGET_COL] = np.select(
        conditions,
        [5, 1],
        default=0,
    ).astype("int8")
    df[GAIN_COL] = np.select(
        conditions,
        [31, 1],
        default=0,
    ).astype("int8")


def validate_train_test_columns(train: pd.DataFrame, test: pd.DataFrame) -> None:
    train_cols = set(train.columns)
    test_cols = set(test.columns)
    missing_from_test = sorted(train_cols - test_cols)
    unexpected_test_only = sorted(test_cols - train_cols)
    leakage_cols = sorted(set(missing_from_test) & TRAIN_ONLY_COLS)
    LOGGER.info("Train-only columns: %s", missing_from_test)
    if leakage_cols:
        LOGGER.info("Marked train-only leakage columns: %s", leakage_cols)
    if unexpected_test_only:
        LOGGER.warning("Columns only in test: %s", unexpected_test_only)
