from __future__ import annotations

from dataclasses import replace
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .config import Config, GROUP_COL, LOGGER, TARGET_COL
from .utils import suffix_path


def make_group_split(
    df: pd.DataFrame, config: Config
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the DataFrame into train and validation sets based on groups defined by group column.
    """
    if not 0.0 < config.valid_fraction < 0.5:
        raise ValueError("--valid-fraction must be between 0 and 0.5")

    def time_aware_split() -> set:
        group_time = (
            df[[GROUP_COL, "date_time"]]
            .groupby(GROUP_COL, sort=False)["date_time"]
            .min()
            .sort_values()
        )
        n_valid = max(1, int(len(group_time) * config.valid_fraction))
        return set(group_time.tail(n_valid).index.to_numpy())

    def random_split() -> set:
        rng = np.random.default_rng(config.seed)
        unique_groups = df[GROUP_COL].drop_duplicates().to_numpy(copy=True)
        rng.shuffle(unique_groups)
        n_valid = max(1, int(len(unique_groups) * config.valid_fraction))
        return set(unique_groups[:n_valid])

    def srch_id_quantile_split() -> set:
        unique_groups = np.sort(df[GROUP_COL].drop_duplicates().to_numpy())
        n_valid = max(1, int(len(unique_groups) * config.valid_fraction))
        return set(unique_groups[-n_valid:])

    if config.split_strategy == "srch_id_quantile":
        LOGGER.info("Creating srch_id-ordered group split")
        valid_groups = srch_id_quantile_split()
    elif config.split_strategy == "time" and "date_time" in df.columns:
        LOGGER.info("Creating time-aware group split")
        valid_groups = time_aware_split()
    else:
        LOGGER.info("Creating random group split")
        valid_groups = random_split()

    valid_mask = df[GROUP_COL].isin(valid_groups)
    train_df = df.loc[~valid_mask].copy()
    valid_df = df.loc[valid_mask].copy()

    train_df.reset_index(drop=True, inplace=True)
    valid_df.reset_index(drop=True, inplace=True)

    LOGGER.info(
        "Train split shape: %s; validation split shape: %s",
        train_df.shape,
        valid_df.shape,
    )

    return train_df, valid_df


def _group_search_month(df: pd.DataFrame) -> pd.Series:
    if "date_time" in df.columns:
        dt = pd.to_datetime(df["date_time"], errors="coerce")
        return dt.dt.month.fillna(0).astype("int16")
    if "search_month" in df.columns:
        return pd.to_numeric(df["search_month"], errors="coerce").fillna(0).astype(
            "int16"
        )
    return pd.Series(np.zeros(len(df), dtype="int16"), index=df.index)


def _fold_summary(groups: pd.Series, folds: np.ndarray) -> Dict[str, Dict[str, int]]:
    return {
        str(fold): {
            "groups": int(groups.iloc[folds == fold].nunique()),
            "rows": int(np.sum(folds == fold)),
        }
        for fold in sorted(np.unique(folds))
    }


def make_stratified_group_folds(
    df: pd.DataFrame,
    n_splits: int,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    Return row-level fold ids while keeping all rows from a srch_id in one fold.

    Stratification is performed at query level with a compact label built from
    random_bool, search month, and whether the query contains a booking. Very
    rare strata are folded into a shared rare bucket; if stratification is still
    not viable, the function falls back to shuffled group folds.
    """
    if GROUP_COL not in df.columns:
        raise ValueError(f"{GROUP_COL} column is required for grouped folds.")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")

    group_frame = pd.DataFrame({GROUP_COL: df[GROUP_COL]})
    if "random_bool" in df.columns:
        group_frame["random_bool"] = (
            pd.to_numeric(df["random_bool"], errors="coerce").fillna(-1).astype("int16")
        )
    else:
        group_frame["random_bool"] = np.int16(-1)
    group_frame["search_month"] = _group_search_month(df)
    if "booking_bool" in df.columns:
        group_frame["query_has_booking"] = (
            pd.to_numeric(df["booking_bool"], errors="coerce")
            .fillna(0)
            .astype("int8")
        )
    else:
        group_frame["query_has_booking"] = np.int8(0)

    group_stats = group_frame.groupby(GROUP_COL, sort=False).agg(
        random_bool=("random_bool", "first"),
        search_month=("search_month", "first"),
        query_has_booking=("query_has_booking", "max"),
    )
    n_groups = len(group_stats)
    if n_groups < 2:
        folds = np.zeros(len(df), dtype="int16")
        return folds, {
            "strategy": "single_fold",
            "n_splits": 1,
            "n_groups": int(n_groups),
            "folds": _fold_summary(df[GROUP_COL], folds),
        }

    n_splits = min(n_splits, n_groups)
    strata = (
        group_stats["random_bool"].astype(str)
        + "_"
        + group_stats["search_month"].astype(str)
        + "_"
        + group_stats["query_has_booking"].astype(str)
    )
    counts = strata.value_counts(sort=False)
    strata = strata.mask(strata.map(counts).lt(n_splits), "rare")

    group_fold_values: np.ndarray
    strategy = "stratified_group"
    if strata.value_counts(sort=False).min() >= n_splits:
        try:
            from sklearn.model_selection import StratifiedKFold

            splitter = StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=seed,
            )
            group_fold_values = np.empty(n_groups, dtype="int16")
            for fold, (_, valid_idx) in enumerate(
                splitter.split(group_stats.index.to_numpy(), strata.to_numpy())
            ):
                group_fold_values[valid_idx] = np.int16(fold)
        except Exception as exc:  # pragma: no cover - fallback depends on sklearn
            LOGGER.warning("Falling back to shuffled group folds: %s", exc)
            strategy = "shuffled_group"
            group_fold_values = _shuffled_group_fold_values(n_groups, n_splits, seed)
    else:
        strategy = "shuffled_group"
        group_fold_values = _shuffled_group_fold_values(n_groups, n_splits, seed)

    fold_map = pd.Series(group_fold_values, index=group_stats.index)
    folds = df[GROUP_COL].map(fold_map).to_numpy(dtype="int16")
    return folds, {
        "strategy": strategy,
        "n_splits": int(n_splits),
        "n_groups": int(n_groups),
        "folds": _fold_summary(df[GROUP_COL], folds),
    }


def _shuffled_group_fold_values(
    n_groups: int,
    n_splits: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    order = np.arange(n_groups, dtype="int32")
    rng.shuffle(order)
    fold_values = np.empty(n_groups, dtype="int16")
    fold_values[order] = (np.arange(n_groups, dtype="int32") % n_splits).astype("int16")
    return fold_values


def dcg_from_relevance(relevance: np.ndarray) -> float:
    if len(relevance) == 0:
        return 0.0
    ranks = np.arange(1, len(relevance) + 1, dtype="float64")
    gains = np.power(2.0, relevance.astype("float64")) - 1.0
    discounts = np.log2(ranks + 1.0)
    return float(np.sum(gains / discounts))


def ndcg_at_k(
    df: pd.DataFrame,
    score_col: str,
    relevance_col: str = TARGET_COL,
    group_col: str = GROUP_COL,
    k: int = 5,
) -> float:
    required = {group_col, score_col, relevance_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Cannot compute NDCG; missing columns: {missing}")

    ordered = df[[group_col, relevance_col, score_col]].sort_values(
        [group_col, score_col], ascending=[True, False], kind="mergesort"
    )
    top = ordered.groupby(group_col, sort=False).head(k).copy()
    top["discount_rank"] = top.groupby(group_col, sort=False).cumcount() + 1
    top["gain"] = np.power(2.0, top[relevance_col].astype("float64")) - 1.0
    top["dcg_part"] = top["gain"] / np.log2(
        top["discount_rank"].astype("float64") + 1.0
    )
    dcg = top.groupby(group_col, sort=False)["dcg_part"].sum()

    ideal = df[[group_col, relevance_col]].sort_values(
        [group_col, relevance_col], ascending=[True, False], kind="mergesort"
    )
    ideal_top = ideal.groupby(group_col, sort=False).head(k).copy()
    ideal_top["discount_rank"] = ideal_top.groupby(group_col, sort=False).cumcount() + 1
    ideal_top["gain"] = np.power(2.0, ideal_top[relevance_col].astype("float64")) - 1.0
    ideal_top["idcg_part"] = ideal_top["gain"] / np.log2(
        ideal_top["discount_rank"].astype("float64") + 1.0
    )
    idcg = ideal_top.groupby(group_col, sort=False)["idcg_part"].sum()

    scores = (dcg / idcg.replace(0, np.nan)).fillna(0.0)
    return float(scores.mean())


def config_for_validation_split(config: Config, split_strategy: str) -> Config:
    suffix = split_strategy.replace("-", "_")
    return replace(
        config,
        split_strategy=split_strategy,
        model_dir=config.model_dir / f"validation_{suffix}",
        metrics_path=suffix_path(config.metrics_path, suffix),
        validation_predictions_path=suffix_path(
            config.validation_predictions_path, suffix
        ),
    )
