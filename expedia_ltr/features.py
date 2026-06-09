from __future__ import annotations

import gc
from typing import (
    Callable,
    ContextManager,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    cast,
)

import numpy as np
import pandas as pd

from .config import (
    AFFINITY_KEY_SPECS,
    BOOKING_COL,
    BUCKET_CROSS_KEY_SPECS,
    BUCKET_SOURCE_SPECS,
    CLICK_COL,
    Config,
    FREQUENCY_KEY_SPECS,
    GROUP_COL,
    HELPER_PREFIXES,
    ITEM_COL,
    LOGGER,
    LOW_CARD_CATEGORICAL_COLS,
    MISSINGNESS_EXCLUDE_COLS,
    MISSING_VALUE_FEATURE_COLS,
    POSITION_AGG_KEY_SPECS,
    PROP_PROFILE_DELTA_COLS,
    PROP_PROFILE_SOURCE_COLS,
    PROP_PROFILE_STATS,
    RELATIVE_FEATURE_BASES,
    SEGMENT_PROFILE_SOURCE_COLS,
    SEGMENT_PROFILE_SPECS,
    TARGET_ENCODING_TARGETS,
    TARGET_KEY_SPECS,
    TRAIN_ONLY_COLS,
    ZERO_AS_MISSING_COLS,
)
from .utils import timer_Factory


def safe_divide(numerator, denominator, fill_value=0.0):
    return (
        numerator.div(denominator)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(fill_value)
        .astype("float32")
    )


def finite_or_zero(value: float | int | np.number | None) -> float:
    if value is None:
        return 0.0
    result = float(value)
    return result if np.isfinite(result) else 0.0


def assign_feature_block(df: pd.DataFrame, columns: Mapping[str, object]) -> None:
    if not columns:
        return
    feature_df = pd.DataFrame(columns, index=df.index, copy=False)
    df[list(feature_df.columns)] = feature_df


def prop_profile_source_columns(
    train_df: pd.DataFrame, apply_df: pd.DataFrame
) -> List[str]:
    """
    Identify numeric columns that are present in both train and apply DFs
    and are not in the TRAIN_ONLY_COLS set or start with any of the HELPER_PREFIXES.
    These columns later be used to create prop_id profile features.
    """
    shared_cols = set(train_df.columns) & set(apply_df.columns)
    exclude = set(TRAIN_ONLY_COLS) | {GROUP_COL, ITEM_COL, "date_time"}
    source_cols: List[str] = []
    for col in PROP_PROFILE_SOURCE_COLS:
        if col not in shared_cols or col in exclude:
            continue
        if any(col.startswith(prefix) for prefix in HELPER_PREFIXES):
            continue
        if pd.api.types.is_numeric_dtype(
            train_df[col]
        ) and pd.api.types.is_numeric_dtype(apply_df[col]):
            source_cols.append(col)
    return source_cols


def attach_profile_features(
    df: pd.DataFrame,
    profile_table: pd.DataFrame,
    feature_names: Sequence[str],
    fill_values: Mapping[str, float],
    chunk_size: int = 24,
) -> None:
    """
    Attach profile features from the profile table to the DataFrame based on the ITEM_COL.
    """
    target_idx = pd.Index(df[ITEM_COL])
    positions = profile_table.index.get_indexer(target_idx)

    missing_mask = positions < 0
    if missing_mask.any():
        positions = positions.copy()
        positions[missing_mask] = 0

    for start in range(0, len(feature_names), chunk_size):
        chunk = list(feature_names[start : start + chunk_size])
        values = profile_table[chunk].to_numpy(dtype="float32", copy=False)[positions]
        if missing_mask.any():
            for idx, feature in enumerate(chunk):
                values[missing_mask, idx] = np.float32(fill_values[feature])
        assign_feature_block(
            df, {feature: values[:, idx] for idx, feature in enumerate(chunk)}
        )


def add_prop_numeric_profile_features(
    train_df: pd.DataFrame,
    apply_df: pd.DataFrame,
) -> List[str]:
    """
    Add numeric profile features for prop_id based on core shared numeric columns.
    in train and apply DataFrames.
    """
    source_cols = prop_profile_source_columns(train_df, apply_df)
    if not source_cols:
        LOGGER.warning(
            "No shared numeric columns available for prop_id profile features"
        )
        return []

    LOGGER.info(
        "Adding prop_id numeric profile features from train+apply rows: %d source columns",
        len(source_cols),
    )
    profile_input = pd.concat(
        [train_df[[ITEM_COL] + source_cols], apply_df[[ITEM_COL] + source_cols]],
        axis=0,
        ignore_index=True,
    )
    grouped = profile_input.groupby(ITEM_COL, sort=False)
    profile_table = grouped[source_cols].agg(PROP_PROFILE_STATS)

    if isinstance(profile_table.columns, pd.MultiIndex):
        profile_table.columns = [
            f"prop_profile_{col}_{stat}"
            for col, stat in profile_table.columns.to_flat_index()
        ]
    else:
        profile_table.columns = [f"prop_profile_{col}" for col in profile_table.columns]

    profile_table = profile_table.astype("float32")
    profile_table["prop_profile_log_count"] = np.log1p(grouped.size()).astype("float32")

    fill_values: Dict[str, float] = {"prop_profile_log_count": 0.0}
    for col in source_cols:
        fill_values[f"prop_profile_{col}_mean"] = finite_or_zero(
            profile_input[col].mean()
        )
        fill_values[f"prop_profile_{col}_median"] = finite_or_zero(
            profile_input[col].median()
        )
        std_col = f"prop_profile_{col}_std"
        if std_col in profile_table.columns:
            fill_values[std_col] = 0.0
    profile_table.fillna(fill_values, inplace=True)

    feature_names = list(profile_table.columns)
    for df in (train_df, apply_df):
        attach_profile_features(df, profile_table, feature_names, fill_values)

        new_columns: Dict[str, object] = {}
        for col in PROP_PROFILE_DELTA_COLS:
            mean_col = f"prop_profile_{col}_mean"
            median_col = f"prop_profile_{col}_median"
            std_col = f"prop_profile_{col}_std"
            if col not in df.columns or mean_col not in df.columns:
                continue
            current = df[col].astype("float32")
            mean_diff = (current - df[mean_col]).astype("float32")
            new_columns[f"{col}_prop_mean_diff"] = mean_diff
            if median_col in df.columns:
                new_columns[f"{col}_prop_median_diff"] = (
                    current - df[median_col]
                ).astype("float32")
            if std_col in df.columns:
                new_columns[f"{col}_prop_zscore"] = safe_divide(
                    mean_diff, df[std_col], fill_value=0.0
                )
        assign_feature_block(df, new_columns)

    del profile_input, profile_table
    gc.collect()
    return feature_names


def add_cross_keys(df: pd.DataFrame) -> None:
    specs = {
        "_key_prop_dest": ("prop_id", "srch_destination_id"),
        "_key_prop_country_visitor_country": (
            "prop_country_id",
            "visitor_location_country_id",
        ),
        "_key_dest_site": ("srch_destination_id", "site_id"),
    }
    new_columns: Dict[str, object] = {}
    for key_name, cols in specs.items():
        if all(col in df.columns for col in cols) and key_name not in df.columns:
            new_columns[key_name] = pd.util.hash_pandas_object(
                df[list(cols)], index=False
            ).astype("uint64")
    assign_feature_block(df, new_columns)


def bucket_edges(series: pd.Series, n_bins: int) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    values = values.dropna().to_numpy(dtype="float64", copy=False)
    if len(values) == 0:
        return np.array([], dtype="float64")
    quantiles = np.linspace(0.0, 1.0, n_bins + 1, dtype="float64")[1:-1]
    return np.unique(np.nanquantile(values, quantiles)).astype("float64")


def apply_bucket_feature(
    df: pd.DataFrame, source_col: str, bucket_col: str, edges: np.ndarray
) -> None:
    values = pd.to_numeric(df[source_col], errors="coerce").to_numpy(dtype="float64")
    missing = ~np.isfinite(values)
    values[missing] = 0.0
    bucket = np.searchsorted(edges, values, side="right").astype("int16")
    bucket[missing] = -1
    df[bucket_col] = bucket


def add_fit_bucket_features(
    fit_df: pd.DataFrame, apply_dfs: Sequence[pd.DataFrame]
) -> List[str]:
    LOGGER.info("Adding train-fitted numeric bucket features")
    feature_names: List[str] = []
    for source_col, bucket_col, n_bins in BUCKET_SOURCE_SPECS:
        if source_col not in fit_df.columns:
            continue
        edges = bucket_edges(fit_df[source_col], n_bins)
        for df in apply_dfs:
            if source_col in df.columns:
                apply_bucket_feature(df, source_col, bucket_col, edges)
        feature_names.append(bucket_col)
    return feature_names


def add_bucket_cross_keys(df: pd.DataFrame) -> None:
    new_columns: Dict[str, object] = {}
    for key_name, cols in BUCKET_CROSS_KEY_SPECS.items():
        if all(col in df.columns for col in cols):
            new_columns[key_name] = pd.util.hash_pandas_object(
                df[list(cols)], index=False
            ).astype("uint64")
    assign_feature_block(df, new_columns)


def add_missingness_features(df: pd.DataFrame) -> None:
    source_cols = [
        col
        for col in df.columns
        if col not in MISSINGNESS_EXCLUDE_COLS
        and not any(col.startswith(prefix) for prefix in HELPER_PREFIXES)
    ]
    if not source_cols:
        df["nans_count"] = np.int16(0)
        return
    df["nans_count"] = df[source_cols].isna().sum(axis=1).astype("int16")


def add_temporal_features(df: pd.DataFrame) -> None:
    if "date_time" not in df.columns:
        return
    dt = pd.to_datetime(df["date_time"], errors="coerce")
    hour = dt.dt.hour
    new_columns: Dict[str, object] = {
        "search_month": dt.dt.month.fillna(0).astype("int8"),
        "search_dayofweek": dt.dt.dayofweek.fillna(-1).astype("int8"),
        "search_hour": hour.fillna(-1).astype("int8"),
        "search_weekofyear": dt.dt.isocalendar().week.fillna(0).astype("int8"),
        "search_dayofyear": dt.dt.dayofyear.fillna(0).astype("int16"),
        "early_night": (hour.ge(20) | hour.lt(3)).astype("int8"),
    }

    if "srch_booking_window" in df.columns:
        booking_window = df["srch_booking_window"].clip(lower=0).fillna(0)
        checkin_dt = dt + pd.to_timedelta(booking_window, unit="D")
        new_columns["checkin_month"] = checkin_dt.dt.month.fillna(0).astype("int8")
        new_columns["checkin_dayofweek"] = checkin_dt.dt.dayofweek.fillna(-1).astype(
            "int8"
        )
    assign_feature_block(df, new_columns)
    df.drop(columns=["date_time"], inplace=True)


def add_trip_and_value_features(df: pd.DataFrame) -> None:
    if "price_usd" in df.columns:
        price = df["price_usd"].clip(lower=0).astype("float32")
        price_log1p = np.log1p(price).astype("float32")
    else:
        price = pd.Series(np.zeros(len(df), dtype="float32"), index=df.index)
        price_log1p = pd.Series(np.zeros(len(df), dtype="float32"), index=df.index)

    stay = (
        df.get("srch_length_of_stay", pd.Series(1, index=df.index))
        .clip(lower=1)
        .astype("float32")
    )
    rooms = (
        df.get("srch_room_count", pd.Series(1, index=df.index))
        .clip(lower=1)
        .astype("float32")
    )
    adults = (
        df.get("srch_adults_count", pd.Series(1, index=df.index))
        .clip(lower=0)
        .astype("float32")
    )
    children = (
        df.get("srch_children_count", pd.Series(0, index=df.index))
        .clip(lower=0)
        .astype("float32")
    )
    people = (adults + children).clip(lower=1)

    new_columns: Dict[str, object] = {
        "price_log1p": price_log1p,
        "total_fee": (price * stay * rooms).astype("float32"),
        "price_per_night": safe_divide(price, stay),
        "price_per_room_night": safe_divide(price, stay * rooms),
        "price_per_person": safe_divide(price, people),
        "people_count": people.astype("float32"),
        "rooms_per_person": safe_divide(rooms, people),
    }

    if "visitor_hist_adr_usd" in df.columns:
        new_columns["hist_adr_price_diff"] = (
            price - df["visitor_hist_adr_usd"]
        ).astype("float32")
        new_columns["hist_adr_price_ratio"] = safe_divide(
            price, df["visitor_hist_adr_usd"]
        )
    if "prop_log_historical_price" in df.columns:
        hist_price = np.expm1(
            df["prop_log_historical_price"].clip(lower=0).astype("float32")
        ).astype("float32")
        hist_price = hist_price.mask(hist_price.le(0))
        new_columns["price_diff_from_historical"] = (
            price - hist_price.fillna(0)
        ).astype("float32")
        new_columns["price_ratio_to_historical"] = safe_divide(price, hist_price)
    if "visitor_hist_starrating" in df.columns and "prop_starrating" in df.columns:
        new_columns["hist_star_diff"] = (
            df["prop_starrating"].astype("float32")
            - df["visitor_hist_starrating"].astype("float32")
        ).astype("float32")

    if "prop_country_id" in df.columns and "visitor_location_country_id" in df.columns:
        new_columns["international_bool"] = (
            df["prop_country_id"].ne(df["visitor_location_country_id"])
        ).astype("int8")

    new_columns["family_trip_bool"] = ((children > 0) | (people >= 4)).astype("int8")
    if "srch_saturday_night_bool" in df.columns:
        saturday = df["srch_saturday_night_bool"].astype("int8")
    else:
        saturday = pd.Series(0, index=df.index, dtype="int8")
    new_columns["business_trip_bool"] = (
        (stay <= 2) & (children == 0) & (people <= 2) & (saturday == 0)
    ).astype("int8")

    star = df.get("prop_starrating", pd.Series(0, index=df.index)).astype("float32")
    review = (
        df.get("prop_review_score", pd.Series(0, index=df.index))
        .fillna(0)
        .astype("float32")
    )
    loc1 = (
        df.get("prop_location_score1", pd.Series(0, index=df.index))
        .fillna(0)
        .astype("float32")
    )
    loc2 = (
        df.get("prop_location_score2", pd.Series(0, index=df.index))
        .fillna(0)
        .astype("float32")
    )
    new_columns["star_review_interaction"] = (star * review).astype("float32")
    new_columns["price_per_star"] = safe_divide(price, star.replace(0, np.nan))
    new_columns["price_per_review"] = safe_divide(price, review.replace(0, np.nan))
    new_columns["value_score"] = safe_divide(
        star + review + loc1 + loc2, price_log1p + 1.0
    )
    assign_feature_block(df, new_columns)


def add_competitor_features(df: pd.DataFrame) -> None:
    rate_cols = [f"comp{i}_rate" for i in range(1, 9) if f"comp{i}_rate" in df.columns]
    inv_cols = [f"comp{i}_inv" for i in range(1, 9) if f"comp{i}_inv" in df.columns]
    pct_cols = [
        f"comp{i}_rate_percent_diff"
        for i in range(1, 9)
        if f"comp{i}_rate_percent_diff" in df.columns
    ]
    new_columns: Dict[str, object] = {}

    if rate_cols:
        rate = df[rate_cols]
        new_columns["comp_rate_missing_count"] = rate.isna().sum(axis=1).astype("int8")
        new_columns["comp_rate_observed_count"] = (
            rate.notna().sum(axis=1).astype("int8")
        )
        new_columns["comp_rate_advantage_count"] = rate.eq(1).sum(axis=1).astype("int8")
        new_columns["comp_rate_disadvantage_count"] = (
            rate.eq(-1).sum(axis=1).astype("int8")
        )
        new_columns["comp_rate_same_count"] = rate.eq(0).sum(axis=1).astype("int8")
        new_columns["comp_rate_mean"] = rate.mean(axis=1).astype("float32")

    if inv_cols:
        inv = df[inv_cols]
        new_columns["comp_inv_missing_count"] = inv.isna().sum(axis=1).astype("int8")
        new_columns["comp_inv_observed_count"] = inv.notna().sum(axis=1).astype("int8")
        new_columns["comp_inv_advantage_count"] = inv.eq(1).sum(axis=1).astype("int8")
        new_columns["comp_inv_disadvantage_count"] = (
            inv.eq(-1).sum(axis=1).astype("int8")
        )
        new_columns["comp_inv_mean"] = inv.mean(axis=1).astype("float32")

    if pct_cols:
        pct = df[pct_cols]
        new_columns["comp_pct_missing_count"] = pct.isna().sum(axis=1).astype("int8")
        new_columns["comp_pct_observed_count"] = pct.notna().sum(axis=1).astype("int8")
        new_columns["comp_pct_mean"] = pct.mean(axis=1).astype("float32")
        new_columns["comp_pct_min"] = pct.min(axis=1).astype("float32")
        new_columns["comp_pct_max"] = pct.max(axis=1).astype("float32")
        new_columns["comp_pct_std"] = pct.std(axis=1).fillna(0).astype("float32")

    missing_cols = rate_cols + inv_cols + pct_cols
    if missing_cols:
        new_columns["comp_total_missing_count"] = (
            df[missing_cols].isna().sum(axis=1).astype("int16")
        )
    assign_feature_block(df, new_columns)


def add_within_search_features(df: pd.DataFrame, chunk_size: int = 6) -> None:
    if GROUP_COL not in df.columns:
        raise ValueError(f"{GROUP_COL} column is required.")

    source_cols = [col for col in RELATIVE_FEATURE_BASES if col in df.columns]
    for start in range(0, len(source_cols), chunk_size):
        chunk = list(source_cols[start : start + chunk_size])
        values = df.loc[:, chunk].astype("float32")
        grouped = values.groupby(df[GROUP_COL], sort=False)
        mean = grouped.transform("mean").astype("float32")
        median = grouped.transform("median").astype("float32")
        min_value = grouped.transform("min").astype("float32")
        max_value = grouped.transform("max").astype("float32")
        std = grouped.transform("std").replace(0, np.nan).astype("float32")
        pct_rank = grouped.rank(method="average", pct=True).astype("float32")
        rank_asc = grouped.rank(method="min", ascending=True).astype("float32")
        rank_desc = grouped.rank(method="min", ascending=False).astype("float32")

        new_columns: Dict[str, object] = {}
        for col in chunk:
            series = values[col]
            mean_diff = (series - mean[col]).astype("float32")
            new_columns[f"{col}_srch_mean_diff"] = mean_diff
            new_columns[f"{col}_srch_median_diff"] = (series - median[col]).astype(
                "float32"
            )
            new_columns[f"{col}_srch_min_diff"] = (series - min_value[col]).astype(
                "float32"
            )
            new_columns[f"{col}_srch_max_diff"] = (series - max_value[col]).astype(
                "float32"
            )
            new_columns[f"{col}_srch_zscore"] = safe_divide(
                mean_diff, std[col], fill_value=0.0
            )
            new_columns[f"{col}_srch_pct_rank"] = pct_rank[col]
            new_columns[f"{col}_srch_rank_asc"] = rank_asc[col]
            new_columns[f"{col}_srch_rank_desc"] = rank_desc[col]
        assign_feature_block(df, new_columns)
        del values, grouped, mean, median, min_value, max_value, std
        del pct_rank, rank_asc, rank_desc
        gc.collect()

    flag_columns: Dict[str, object] = {}
    if "price_usd_srch_rank_asc" in df.columns:
        flag_columns["price_order"] = df["price_usd_srch_rank_asc"].astype("float32")
        flag_columns["price_is_cheapest"] = (
            df["price_usd_srch_rank_asc"].eq(1).astype("int8")
        )
    if "price_usd_srch_rank_desc" in df.columns:
        flag_columns["price_is_most_expensive"] = (
            df["price_usd_srch_rank_desc"].eq(1).astype("int8")
        )
    if "prop_review_score_srch_rank_desc" in df.columns:
        flag_columns["review_is_best_in_search"] = (
            df["prop_review_score_srch_rank_desc"].eq(1).astype("int8")
        )
    if "value_score_srch_rank_desc" in df.columns:
        flag_columns["value_is_best_in_search"] = (
            df["value_score_srch_rank_desc"].eq(1).astype("int8")
        )
    assign_feature_block(df, flag_columns)


def add_common_features(
    df: pd.DataFrame,
    label: str,
    timer: Callable[[str], ContextManager[None]],
) -> None:
    with timer(f"{label}: missingness features"):
        add_missingness_features(df)
    with timer(f"{label}: cross keys"):
        add_cross_keys(df)
    with timer(f"{label}: temporal features"):
        add_temporal_features(df)
    with timer(f"{label}: trip and value features"):
        add_trip_and_value_features(df)
    with timer(f"{label}: competitor features"):
        add_competitor_features(df)
    with timer(f"{label}: within-search features"):
        add_within_search_features(df)


def make_group_folds(groups: pd.Series, n_splits: int, seed: int) -> np.ndarray:
    unique_groups = pd.Series(groups.drop_duplicates().to_numpy())
    if len(unique_groups) < 2 or n_splits < 2:
        return np.zeros(len(groups), dtype="int16")
    n_splits = min(n_splits, len(unique_groups))
    shuffled = unique_groups.sample(frac=1.0, random_state=seed).to_numpy()
    fold_values = np.arange(len(shuffled), dtype="int32") % n_splits
    fold_map = pd.Series(fold_values, index=shuffled)
    return groups.map(fold_map).to_numpy(dtype="int16")


def fit_target_encoding_table(
    fit_df: pd.DataFrame,
    key: str,
    alias: str,
    smoothing: float,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    required = [key] + [target for target, _ in TARGET_ENCODING_TARGETS]
    missing = [col for col in required if col not in fit_df.columns]
    if missing:
        raise ValueError(f"Cannot target encode {key}; missing columns: {missing}")

    priors = {
        target: float(fit_df[target].mean()) for target, _ in TARGET_ENCODING_TARGETS
    }
    agg_kwargs = {
        f"{short}_sum": (target, "sum") for target, short in TARGET_ENCODING_TARGETS
    }
    agg_kwargs["count"] = (key, "size")
    stats = fit_df.groupby(key, sort=False).agg(**agg_kwargs)

    table = pd.DataFrame(index=stats.index)
    count = stats["count"].astype("float32")
    for target, short in TARGET_ENCODING_TARGETS:
        encoded = (
            stats[f"{short}_sum"].astype("float32") + priors[target] * smoothing
        ) / (count + smoothing)
        table[f"{alias}_{short}_te"] = encoded.astype("float32")
    booking_per_click_prior = safe_divide(
        pd.Series([priors[BOOKING_COL]], dtype="float32"),
        pd.Series([priors[CLICK_COL]], dtype="float32"),
    ).iloc[0]
    table[f"{alias}_booking_per_click_te"] = (
        (
            stats["booking_sum"].astype("float32")
            + np.float32(booking_per_click_prior * smoothing)
        )
        / (stats["click_sum"].astype("float32") + np.float32(smoothing))
    ).astype("float32")
    table[f"{alias}_te_count"] = stats["count"].astype("int32")
    return table, priors


def apply_target_encoding_table(
    df: pd.DataFrame,
    key: str,
    alias: str,
    table: pd.DataFrame,
    priors: Mapping[str, float],
) -> List[str]:
    feature_names: List[str] = []
    new_columns: Dict[str, object] = {}
    for target, short in TARGET_ENCODING_TARGETS:
        col = f"{alias}_{short}_te"
        new_columns[col] = (
            df[key].map(table[col]).fillna(priors[target]).astype("float32")
        )
        feature_names.append(col)

    booking_per_click_col = f"{alias}_booking_per_click_te"
    booking_per_click_prior = safe_divide(
        pd.Series([priors[BOOKING_COL]], dtype="float32"),
        pd.Series([priors[CLICK_COL]], dtype="float32"),
    ).iloc[0]
    new_columns[booking_per_click_col] = (
        df[key]
        .map(table[booking_per_click_col])
        .fillna(booking_per_click_prior)
        .astype("float32")
    )
    feature_names.append(booking_per_click_col)

    raw_count = df[key].map(table[f"{alias}_te_count"]).fillna(0).astype("float32")
    count_col = f"{alias}_te_log_count"
    new_columns[count_col] = np.log1p(raw_count).astype("float32")
    feature_names.append(count_col)
    assign_feature_block(df, new_columns)
    return feature_names


def position_feature_names(alias: str) -> List[str]:
    return [
        f"{alias}_position_mean",
        f"{alias}_position_median",
        f"{alias}_top5_rate",
        f"{alias}_reciprocal_position_mean",
        f"{alias}_position_count_log",
    ]


def position_fit_source(fit_df: pd.DataFrame) -> pd.DataFrame:
    if "position" not in fit_df.columns:
        raise ValueError("position is required for historical position features.")
    base = fit_df.loc[fit_df["position"].notna()]
    if "random_bool" not in base.columns:
        return base
    randomized_filter = base["random_bool"].eq(0)
    if randomized_filter.any():
        return base.loc[randomized_filter]
    return base


def fit_position_aggregate_table(
    fit_df: pd.DataFrame,
    key: str,
    alias: str,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    source = position_fit_source(fit_df)
    feature_names = position_feature_names(alias)
    mean_col, median_col, top5_col, reciprocal_col, count_col = feature_names

    if source.empty:
        priors = {
            mean_col: 0.0,
            median_col: 0.0,
            top5_col: 0.0,
            reciprocal_col: 0.0,
            count_col: 0.0,
        }
        return pd.DataFrame(columns=feature_names), priors

    position = pd.to_numeric(source["position"], errors="coerce").astype("float32")
    priors = {
        mean_col: finite_or_zero(position.mean()),
        median_col: finite_or_zero(position.median()),
        top5_col: finite_or_zero(position.le(5).mean()),
        reciprocal_col: finite_or_zero((1.0 / position.clip(lower=1)).mean()),
        count_col: 0.0,
    }

    stats_input = pd.DataFrame(
        {
            key: source[key],
            "position": position,
            "_position_top5": position.le(5).astype("float32"),
            "_reciprocal_position": (1.0 / position.clip(lower=1)).astype("float32"),
        },
        index=source.index,
    )
    stats = stats_input.groupby(key, sort=False).agg(
        **{
            mean_col: ("position", "mean"),
            median_col: ("position", "median"),
            top5_col: ("_position_top5", "mean"),
            reciprocal_col: ("_reciprocal_position", "mean"),
            "_position_count": ("position", "size"),
        }
    )
    stats[count_col] = np.log1p(stats["_position_count"].astype("float32")).astype(
        "float32"
    )
    table = stats[feature_names].astype("float32")
    del stats_input, stats
    return table, priors


def position_aggregate_values(
    df: pd.DataFrame,
    key: str,
    alias: str,
    table: pd.DataFrame,
    priors: Mapping[str, float],
) -> Dict[str, object]:
    key_values = df[key]
    values: Dict[str, object] = {}
    for feature in position_feature_names(alias):
        fill_value = 0.0 if feature.endswith("_count_log") else priors[feature]
        if table.empty:
            values[feature] = np.full(len(df), np.float32(fill_value), dtype="float32")
        else:
            values[feature] = (
                key_values.map(table[feature])
                .fillna(fill_value)
                .astype("float32")
            )
    return values


def add_oof_position_aggregates(
    train_df: pd.DataFrame,
    config: Config,
) -> List[str]:
    if "position" not in train_df.columns:
        LOGGER.warning("Skipping historical position aggregates; position is missing.")
        return []

    LOGGER.info("Adding out-of-fold historical position aggregates")
    folds = make_group_folds(
        train_df[GROUP_COL], config.target_encoding_folds, config.seed
    )
    n_folds = int(folds.max()) + 1
    if n_folds < 2:
        LOGGER.warning(
            "Only one fold available; historical position aggregates fall back to global priors."
        )

    feature_names: List[str] = []
    for key, alias in POSITION_AGG_KEY_SPECS:
        if key not in train_df.columns:
            continue
        key_features = position_feature_names(alias)
        feature_names.extend(key_features)
        cols_needed = list(
            dict.fromkeys(
                [key, "position"]
                + (["random_bool"] if "random_bool" in train_df.columns else [])
            )
        )
        table, priors = fit_position_aggregate_table(
            train_df[cols_needed], key, alias
        )
        assign_feature_block(
            train_df,
            position_aggregate_values(train_df, key, alias, table, priors),
        )

        if n_folds >= 2:
            for fold in range(n_folds):
                holdout_mask = folds == fold
                fit_part = cast(pd.DataFrame, train_df.loc[~holdout_mask, cols_needed])
                fold_table, fold_priors = fit_position_aggregate_table(
                    fit_part, key, alias
                )
                holdout_index = train_df.index[holdout_mask]
                holdout_values = position_aggregate_values(
                    train_df.loc[holdout_index], key, alias, fold_table, fold_priors
                )
                for feature, values in holdout_values.items():
                    train_df.loc[holdout_index, feature] = np.asarray(
                        values, dtype="float32"
                    )
                del fit_part, fold_table, holdout_values
                gc.collect()
        del table
    return sorted(set(feature_names))


def add_fit_position_aggregates(
    fit_df: pd.DataFrame,
    apply_dfs: Sequence[pd.DataFrame],
) -> List[str]:
    if "position" not in fit_df.columns:
        LOGGER.warning("Skipping fit/apply position aggregates; position is missing.")
        return []

    LOGGER.info("Applying historical position aggregates fitted on training rows")
    feature_names: List[str] = []
    for key, alias in POSITION_AGG_KEY_SPECS:
        if key not in fit_df.columns:
            continue
        cols_needed = list(
            dict.fromkeys(
                [key, "position"]
                + (["random_bool"] if "random_bool" in fit_df.columns else [])
            )
        )
        table, priors = fit_position_aggregate_table(
            fit_df[cols_needed], key, alias
        )
        for df in apply_dfs:
            if key in df.columns:
                assign_feature_block(
                    df, position_aggregate_values(df, key, alias, table, priors)
                )
        feature_names.extend(position_feature_names(alias))
        del table
        gc.collect()
    return sorted(set(feature_names))


def add_oof_target_encodings(
    train_df: pd.DataFrame,
    config: Config,
) -> List[str]:
    LOGGER.info("Adding out-of-fold target encodings")
    feature_names: List[str] = []
    folds = make_group_folds(
        train_df[GROUP_COL], config.target_encoding_folds, config.seed
    )
    n_folds = int(folds.max()) + 1
    if n_folds < 2:
        LOGGER.warning(
            "Only one fold available; target encodings fall back to smoothed priors."
        )

    for key, alias in TARGET_KEY_SPECS:
        if key not in train_df.columns:
            continue
        LOGGER.info("OOF target encoding key=%s", key)
        priors = {
            target: float(train_df[target].mean())
            for target, _ in TARGET_ENCODING_TARGETS
        }
        key_features = [
            f"{alias}_{short}_te" for _, short in TARGET_ENCODING_TARGETS
        ] + [f"{alias}_booking_per_click_te", f"{alias}_te_log_count"]
        new_columns: Dict[str, object] = {}
        for target, short in TARGET_ENCODING_TARGETS:
            new_columns[f"{alias}_{short}_te"] = np.full(
                len(train_df), np.float32(priors[target]), dtype="float32"
            )
        booking_per_click_prior = safe_divide(
            pd.Series([priors[BOOKING_COL]], dtype="float32"),
            pd.Series([priors[CLICK_COL]], dtype="float32"),
        ).iloc[0]
        new_columns[f"{alias}_booking_per_click_te"] = np.full(
            len(train_df), np.float32(booking_per_click_prior), dtype="float32"
        )
        new_columns[f"{alias}_te_log_count"] = np.zeros(len(train_df), dtype="float32")
        assign_feature_block(train_df, new_columns)

        if n_folds >= 2:
            cols_needed = [key] + [target for target, _ in TARGET_ENCODING_TARGETS]
            for fold in range(n_folds):
                holdout_mask = folds == fold
                fit_part = cast(pd.DataFrame, train_df.loc[~holdout_mask, cols_needed])
                table, fold_priors = fit_target_encoding_table(
                    fit_part, key, alias, config.target_encoding_smoothing
                )
                holdout_index = train_df.index[holdout_mask]
                key_values = train_df.loc[holdout_index, key]
                for target, short in TARGET_ENCODING_TARGETS:
                    col = f"{alias}_{short}_te"
                    train_df.loc[holdout_index, col] = (
                        key_values.map(table[col])
                        .fillna(fold_priors[target])
                        .astype("float32")
                    )
                booking_per_click_col = f"{alias}_booking_per_click_te"
                fold_booking_per_click_prior = safe_divide(
                    pd.Series([fold_priors[BOOKING_COL]], dtype="float32"),
                    pd.Series([fold_priors[CLICK_COL]], dtype="float32"),
                ).iloc[0]
                train_df.loc[holdout_index, booking_per_click_col] = (
                    key_values.map(table[booking_per_click_col])
                    .fillna(fold_booking_per_click_prior)
                    .astype("float32")
                )
                raw_count = (
                    key_values.map(table[f"{alias}_te_count"])
                    .fillna(0)
                    .astype("float32")
                )
                train_df.loc[holdout_index, f"{alias}_te_log_count"] = np.log1p(
                    raw_count
                ).astype("float32")
                del fit_part, table
                gc.collect()
        feature_names.extend(key_features)
    return feature_names


def add_fit_target_encodings(
    fit_df: pd.DataFrame,
    apply_dfs: Sequence[pd.DataFrame],
    config: Config,
) -> List[str]:
    LOGGER.info("Applying target encodings fitted on the current training portion")
    feature_names: List[str] = []
    for key, alias in TARGET_KEY_SPECS:
        if key not in fit_df.columns:
            continue
        LOGGER.info("Fit/apply target encoding key=%s", key)
        cols_needed = [key] + [target for target, _ in TARGET_ENCODING_TARGETS]
        table, priors = fit_target_encoding_table(
            fit_df[cols_needed], key, alias, config.target_encoding_smoothing
        )
        for df in apply_dfs:
            if key in df.columns:
                feature_names.extend(
                    apply_target_encoding_table(df, key, alias, table, priors)
                )
        del table
        gc.collect()
    return sorted(set(feature_names))


def add_frequency_features(
    fit_df: pd.DataFrame, apply_dfs: Sequence[pd.DataFrame]
) -> List[str]:
    LOGGER.info("Adding train-fitted frequency features")
    feature_names: List[str] = []
    for key, alias in FREQUENCY_KEY_SPECS:
        if key not in fit_df.columns:
            continue
        counts = fit_df.groupby(key, sort=False).size().astype("int32")
        col = f"{alias}_log_frequency"
        for df in apply_dfs:
            if key in df.columns:
                assign_feature_block(
                    df,
                    {
                        col: np.log1p(
                            df[key].map(counts).fillna(0).astype("float32")
                        ).astype("float32")
                    },
                )
        feature_names.append(col)
        del counts
    return feature_names


def cross_hash(df: pd.DataFrame, cols: Sequence[str]) -> pd.Series:
    return pd.util.hash_pandas_object(df[list(cols)], index=False).astype("uint64")


def add_context_affinity_features(
    fit_df: pd.DataFrame, apply_dfs: Sequence[pd.DataFrame]
) -> List[str]:
    LOGGER.info("Adding train-fitted context affinity features")
    feature_names: List[str] = []
    total_rows = max(1, len(fit_df))
    for alias, left_col, right_col in AFFINITY_KEY_SPECS:
        if left_col not in fit_df.columns or right_col not in fit_df.columns:
            continue

        left_counts = fit_df.groupby(left_col, sort=False).size().astype("float32")
        right_counts = fit_df.groupby(right_col, sort=False).size().astype("float32")
        fit_key = cross_hash(fit_df, [left_col, right_col])
        cross_counts = fit_key.value_counts(sort=False).astype("float32")

        cols = [
            f"{alias}_log_count",
            f"{alias}_left_share",
            f"{alias}_right_share",
            f"{alias}_lift",
        ]
        feature_names.extend(cols)

        for df in apply_dfs:
            if left_col not in df.columns or right_col not in df.columns:
                continue
            apply_key = cross_hash(df, [left_col, right_col])
            cross = apply_key.map(cross_counts).fillna(0).astype("float32")
            left = df[left_col].map(left_counts).fillna(0).astype("float32")
            right = df[right_col].map(right_counts).fillna(0).astype("float32")
            expected = ((left * right) / np.float32(total_rows)).astype("float32")

            assign_feature_block(
                df,
                {
                    f"{alias}_log_count": np.log1p(cross).astype("float32"),
                    f"{alias}_left_share": safe_divide(cross, left),
                    f"{alias}_right_share": safe_divide(cross, right),
                    f"{alias}_lift": safe_divide(cross + 1.0, expected + 1.0),
                },
            )

        del left_counts, right_counts, fit_key, cross_counts
        gc.collect()

    return sorted(set(feature_names))


def cleaned_missing_series(df: pd.DataFrame, col: str) -> pd.Series:
    series = pd.to_numeric(df[col], errors="coerce").astype("float32")
    if col in ZERO_AS_MISSING_COLS:
        series = series.mask(series.eq(0))
    return series


def add_fit_missing_value_features(
    fit_df: pd.DataFrame, apply_dfs: Sequence[pd.DataFrame]
) -> List[str]:
    LOGGER.info("Adding train-fitted missing-value variant features")
    feature_names: List[str] = []
    segment_col = "prop_country_id"

    for col in MISSING_VALUE_FEATURE_COLS:
        if col not in fit_df.columns:
            continue
        clean_fit = cleaned_missing_series(fit_df, col)
        global_median = finite_or_zero(clean_fit.median())
        observed_min = finite_or_zero(clean_fit.min())
        observed_std = finite_or_zero(clean_fit.std())
        sentinel = np.float32(observed_min - max(1.0, observed_std))
        segment_medians: Optional[pd.Series] = None
        if segment_col in fit_df.columns:
            segment_medians = clean_fit.groupby(
                fit_df[segment_col], sort=False
            ).median()

        flag_col = f"{col}_missing_or_zero"
        imputed_col = f"{col}_country_imputed"
        sentinel_col = f"{col}_missing_worst_sentinel"
        diff_col = f"{col}_country_median_diff"
        clean_col: Optional[str] = None
        zero_col: Optional[str] = None
        feature_names.extend([flag_col, imputed_col, sentinel_col, diff_col])
        if col in ZERO_AS_MISSING_COLS:
            clean_col = f"{col}_zero_clean"
            zero_col = f"{col}_zero_flag"
            feature_names.extend([clean_col, zero_col])

        for df in apply_dfs:
            if col not in df.columns:
                continue
            clean = cleaned_missing_series(df, col)
            missing = clean.isna()
            if segment_medians is not None and segment_col in df.columns:
                mapped_median = df[segment_col].map(segment_medians)
            else:
                mapped_median = pd.Series(np.nan, index=df.index, dtype="float32")
            imputed = (
                clean.fillna(mapped_median).fillna(global_median).astype("float32")
            )

            new_columns = {
                flag_col: missing.astype("int8"),
                imputed_col: imputed,
                sentinel_col: clean.fillna(sentinel).astype("float32"),
                diff_col: (clean - imputed).fillna(0).astype("float32"),
            }
            if clean_col is not None and zero_col is not None:
                raw = pd.to_numeric(df[col], errors="coerce").astype("float32")
                new_columns[clean_col] = clean.astype("float32")
                new_columns[zero_col] = raw.eq(0).fillna(False).astype("int8")
            assign_feature_block(df, new_columns)

        del clean_fit, segment_medians
        gc.collect()

    return sorted(set(feature_names))


def add_segment_profile_features(
    fit_df: pd.DataFrame, apply_dfs: Sequence[pd.DataFrame]
) -> List[str]:
    LOGGER.info("Adding train-fitted segment-normalized features")
    feature_names: List[str] = []
    for alias, key_col in SEGMENT_PROFILE_SPECS:
        if key_col not in fit_df.columns:
            continue
        source_cols = [
            col
            for col in SEGMENT_PROFILE_SOURCE_COLS
            if col in fit_df.columns and pd.api.types.is_numeric_dtype(fit_df[col])
        ]
        if not source_cols:
            continue

        grouped = fit_df.groupby(key_col, sort=False)[source_cols]
        means = grouped.mean().astype("float32")
        stds = grouped.std().replace(0, np.nan).astype("float32")
        global_means = fit_df[source_cols].mean().astype("float32")
        global_stds = (
            fit_df[source_cols].std().replace(0, np.nan).fillna(1.0).astype("float32")
        )

        for df in apply_dfs:
            if key_col not in df.columns:
                continue
            new_columns: Dict[str, object] = {}
            for col in source_cols:
                segment_mean = (
                    df[key_col]
                    .map(means[col])
                    .fillna(float(global_means[col]))
                    .astype("float32")
                )
                segment_std = (
                    df[key_col]
                    .map(stds[col])
                    .fillna(float(global_stds[col]))
                    .replace(0, np.nan)
                    .astype("float32")
                )
                diff_col = f"{alias}_{col}_segment_mean_diff"
                z_col = f"{alias}_{col}_segment_zscore"
                diff = (df[col].astype("float32") - segment_mean).astype("float32")
                new_columns[diff_col] = diff
                new_columns[z_col] = safe_divide(diff, segment_std)
                feature_names.extend([diff_col, z_col])
            assign_feature_block(df, new_columns)

        del grouped, means, stds
        gc.collect()

    return sorted(set(feature_names))


def build_ranking_features(
    train_df: pd.DataFrame,
    apply_df: pd.DataFrame,
    config: Config,
) -> Tuple[List[str], List[str]]:
    """
    Build features with leakage safe encodings and return the list of feature names to use for modeling.
    """
    timer = timer_Factory(config.profile)

    with timer("feature stage: prop numeric profiles"):
        add_prop_numeric_profile_features(train_df, apply_df)
    LOGGER.info("Building common features for train portion")
    add_common_features(train_df, "train common", timer)
    LOGGER.info("Building common features for apply portion")
    add_common_features(apply_df, "apply common", timer)

    with timer("feature stage: missing value variants"):
        add_fit_missing_value_features(train_df, [train_df, apply_df])
    with timer("feature stage: numeric buckets"):
        add_fit_bucket_features(train_df, [train_df, apply_df])
    with timer("feature stage: bucket cross keys"):
        add_bucket_cross_keys(train_df)
        add_bucket_cross_keys(apply_df)
    with timer("feature stage: segment profiles"):
        add_segment_profile_features(train_df, [train_df, apply_df])
    with timer("feature stage: OOF historical position aggregates"):
        add_oof_position_aggregates(train_df, config)
    with timer("feature stage: fit/apply historical position aggregates"):
        add_fit_position_aggregates(train_df, [apply_df])
    with timer("feature stage: frequency features"):
        add_frequency_features(train_df, [train_df, apply_df])
    with timer("feature stage: context affinity features"):
        add_context_affinity_features(train_df, [train_df, apply_df])
    with timer("feature stage: OOF target encodings"):
        add_oof_target_encodings(train_df, config)
    with timer("feature stage: fit/apply target encodings"):
        add_fit_target_encodings(train_df, [apply_df], config)
    gc.collect()  # Force garbage collection before selecting features to reduce memory usage

    with timer("feature stage: select feature columns"):
        features = select_feature_columns(train_df, apply_df, config)
    categorical_features = [
        col
        for col in features
        if col in LOW_CARD_CATEGORICAL_COLS
        or col.endswith("_bucket")
        or (config.use_raw_prop_id and col == ITEM_COL)
    ]
    LOGGER.info(
        "Prepared %d features; %d categorical", len(features), len(categorical_features)
    )
    return features, categorical_features


def select_feature_columns(
    train_df: pd.DataFrame,
    apply_df: pd.DataFrame,
    config: Config,
) -> List[str]:
    apply_cols = set(apply_df.columns)
    exclude = set(TRAIN_ONLY_COLS) | {GROUP_COL, "date_time"}
    if not config.use_raw_prop_id:
        exclude.add(ITEM_COL)

    features: List[str] = []
    for col in train_df.columns:
        if col not in apply_cols or col in exclude:
            continue
        if any(col.startswith(prefix) for prefix in HELPER_PREFIXES):
            continue
        if pd.api.types.is_numeric_dtype(train_df[col]):
            features.append(col)
    return sorted(features)
