from __future__ import annotations

import gc
import importlib.util
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, cast

import lightgbm as lgb
import numpy as np
import pandas as pd

from .artifacts import save_pickle, write_json
from .config import (
    BOOKING_COL,
    CLICK_COL,
    Config,
    GROUP_COL,
    ITEM_COL,
    LOGGER,
    LOW_CARD_CATEGORICAL_COLS,
    OOFModelName,
    TARGET_COL,
)
from .data import add_relevance, load_dataset, validate_train_test_columns
from .features import build_ranking_features
from .model import (
    fit_classifier,
    fit_ranker,
    predict_classifier,
    predict_ranker,
    sort_for_ranker,
)
from .training import add_auxiliary_position_feature
from .utils import timer_Factory
from .validation import make_stratified_group_folds, ndcg_at_k

CATBOOST_RAW_CATEGORICAL_COLS = (
    ITEM_COL,
    "srch_destination_id",
    "site_id",
    "visitor_location_country_id",
    "prop_country_id",
    "prop_brand_bool",
    "promotion_flag",
    "random_bool",
    "srch_saturday_night_bool",
)

META_CONTEXT_COLS = ("random_bool", "search_month")


def score_col_for_model(model_name: OOFModelName) -> str:
    return f"{model_name}_score"


def predictions_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_test_predictions.csv")


def optional_dependency_available(model_name: OOFModelName) -> bool:
    if model_name.startswith("xgb"):
        return importlib.util.find_spec("xgboost") is not None
    if model_name.startswith("catboost"):
        return importlib.util.find_spec("catboost") is not None
    return True


def available_oof_models(config: Config) -> Tuple[OOFModelName, ...]:
    selected: List[OOFModelName] = []
    for model_name in config.oof_models:
        if optional_dependency_available(model_name):
            selected.append(model_name)
        else:
            LOGGER.warning(
                "Skipping OOF model %s because its optional dependency is unavailable",
                model_name,
            )
    if not selected:
        raise ValueError("No OOF base models are available after dependency checks.")
    return tuple(selected)


def assert_lgbm_feature_policy(
    features: Sequence[str],
    categorical_features: Sequence[str],
) -> None:
    if ITEM_COL in features or ITEM_COL in categorical_features:
        raise ValueError(
            "Raw prop_id must not be passed to LightGBM. Use encoded prop_id "
            "features for LightGBM and CatBoost for raw high-cardinality IDs."
        )


def integer_weight_compositions(total: int, parts: int):
    if parts == 1:
        yield (total,)
        return
    for value in range(total + 1):
        for rest in integer_weight_compositions(total - value, parts - 1):
            yield (value, *rest)


def blend_weight_grid(weight_count: int, step: float) -> List[Tuple[float, ...]]:
    units = max(1, int(round(1.0 / step)))
    return [
        tuple(value / units for value in values)
        for values in integer_weight_compositions(units, weight_count)
    ]


def rank_average_matrix(df: pd.DataFrame, score_cols: Sequence[str]) -> np.ndarray:
    return np.vstack(
        [
            df.groupby(GROUP_COL, sort=False)[col]
            .rank(method="average", pct=True)
            .astype("float32")
            .to_numpy()
            for col in score_cols
        ]
    )


def weighted_rank_average(
    df: pd.DataFrame,
    score_cols: Sequence[str],
    weights: Sequence[float],
) -> np.ndarray:
    matrix = rank_average_matrix(df, score_cols)
    return np.average(
        matrix,
        axis=0,
        weights=np.asarray(weights, dtype="float32"),
    ).astype("float32")


def tune_rank_average_weights(
    df: pd.DataFrame,
    score_cols: Sequence[str],
    step: float,
) -> Tuple[Tuple[float, ...], float, np.ndarray]:
    if len(score_cols) == 1:
        score = ndcg_at_k(df, score_cols[0])
        return (1.0,), score, df[score_cols[0]].to_numpy(dtype="float32")

    matrix = rank_average_matrix(df, score_cols)
    candidate_col = "_rank_average_candidate"
    best_score = -np.inf
    best_weights: Tuple[float, ...] = ()
    best_preds = np.zeros(len(df), dtype="float32")

    for weights in blend_weight_grid(len(score_cols), step):
        preds = np.average(
            matrix,
            axis=0,
            weights=np.asarray(weights, dtype="float32"),
        ).astype("float32")
        df[candidate_col] = preds
        score = ndcg_at_k(df, candidate_col)
        if score > best_score:
            best_score = score
            best_weights = weights
            best_preds = preds

    df.drop(columns=[candidate_col], inplace=True)
    return best_weights, float(best_score), best_preds


def add_score_rank_features(df: pd.DataFrame, score_cols: Sequence[str]) -> List[str]:
    rank_cols: List[str] = []
    for score_col in score_cols:
        rank_col = f"{score_col}_rank_pct"
        df[rank_col] = (
            df.groupby(GROUP_COL, sort=False)[score_col]
            .rank(method="average", pct=True)
            .astype("float32")
        )
        rank_cols.append(rank_col)
    return rank_cols


def meta_candidate_columns(
    df: pd.DataFrame,
    score_cols: Sequence[str],
    feature_cols: Sequence[str],
) -> List[str]:
    rank_cols = add_score_rank_features(df, score_cols)
    count_cols = [
        col
        for col in feature_cols
        if col in df.columns
        and (
            col.endswith("_te_log_count")
            or col.endswith("_position_count_log")
            or col.endswith("_log_frequency")
        )
    ]
    context_cols = [col for col in META_CONTEXT_COLS if col in df.columns]
    return sorted(set([*score_cols, *rank_cols, *count_cols, *context_cols]))


def align_meta_features(df: pd.DataFrame, feature_cols: Sequence[str]) -> None:
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.float32(0.0)


def meta_categorical_features(feature_cols: Sequence[str]) -> List[str]:
    return [
        col
        for col in feature_cols
        if col in LOW_CARD_CATEGORICAL_COLS or col in META_CONTEXT_COLS
    ]


def small_meta_config(config: Config) -> Config:
    return replace(
        config,
        n_estimators=min(config.n_estimators, 500),
        learning_rate=max(config.learning_rate, 0.05),
        num_leaves=min(config.num_leaves, 31),
        min_child_samples=max(200, min(config.min_child_samples, 2000)),
        colsample_bytree=min(1.0, max(config.colsample_bytree, 0.9)),
    )


def fit_meta_ranker_oof(
    meta_df: pd.DataFrame,
    feature_cols: Sequence[str],
    config: Config,
) -> Tuple[np.ndarray, float]:
    preds = np.zeros(len(meta_df), dtype="float32")
    meta_config = small_meta_config(config)
    categorical = meta_categorical_features(feature_cols)

    for fold in sorted(meta_df["_oof_fold"].unique()):
        valid_mask = meta_df["_oof_fold"].eq(fold)
        train_part = meta_df.loc[~valid_mask].copy()
        valid_part = meta_df.loc[valid_mask].copy()
        model = fit_ranker(
            train_part,
            valid_part,
            feature_cols,
            categorical,
            meta_config,
        )
        preds[valid_mask.to_numpy()] = predict_ranker(model, valid_part, feature_cols)
        del train_part, valid_part, model
        gc.collect()

    score_df = meta_df[[GROUP_COL, TARGET_COL]].copy()
    score_df["meta_ranker_score"] = preds
    return preds, ndcg_at_k(score_df, "meta_ranker_score")


def fit_final_meta_ranker(
    meta_df: pd.DataFrame,
    test_meta_df: pd.DataFrame,
    feature_cols: Sequence[str],
    config: Config,
) -> Tuple[lgb.LGBMRanker, np.ndarray]:
    meta_config = small_meta_config(config)
    categorical = meta_categorical_features(feature_cols)
    model = fit_ranker(meta_df, None, feature_cols, categorical, meta_config)
    preds = predict_ranker(model, test_meta_df, feature_cols)
    return model, preds


def segment_scores(
    df: pd.DataFrame,
    score_cols: Sequence[str],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    result: Dict[str, Dict[str, Dict[str, float]]] = {}
    segment_df = df.copy()
    if BOOKING_COL in segment_df.columns:
        segment_df["query_has_booking"] = (
            segment_df.groupby(GROUP_COL, sort=False)[BOOKING_COL]
            .transform("max")
            .astype("int8")
        )
    if "date_time" in segment_df.columns:
        segment_df["search_month"] = (
            pd.to_datetime(segment_df["date_time"], errors="coerce")
            .dt.month.fillna(0)
            .astype("int16")
        )

    for segment_col in ("random_bool", "search_month", "query_has_booking"):
        if segment_col not in segment_df.columns:
            continue
        result[segment_col] = {}
        for value, part in segment_df.groupby(segment_col, sort=True):
            if part.empty:
                continue
            result[segment_col][str(value)] = {
                score_col: ndcg_at_k(part, score_col) for score_col in score_cols
            }
    return result


def fit_xgb_ranker(
    train_df: pd.DataFrame,
    valid_df: Optional[pd.DataFrame],
    features: Sequence[str],
    config: Config,
    n_estimators: Optional[int] = None,
) -> object:
    from xgboost import XGBRanker

    train_sorted, train_group = sort_for_ranker(train_df, features)
    max_depth = config.max_depth if config.max_depth > 0 else 8
    model_kwargs: Dict[str, object] = {}
    if valid_df is not None and len(valid_df) > 0:
        model_kwargs["early_stopping_rounds"] = config.early_stopping_rounds
    model = XGBRanker(
        objective="rank:ndcg",
        eval_metric="ndcg@5",
        tree_method="hist",
        n_estimators=n_estimators or config.n_estimators,
        learning_rate=config.learning_rate,
        max_depth=max_depth,
        min_child_weight=max(1.0, config.min_child_samples / 1000.0),
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        reg_alpha=config.reg_alpha,
        reg_lambda=config.reg_lambda,
        random_state=config.seed + 31,
        n_jobs=config.n_jobs,
        **model_kwargs,
    )
    fit_kwargs: Dict[str, object] = {
        "X": train_sorted.loc[:, list(features)],
        "y": train_sorted[TARGET_COL].to_numpy(dtype="float32"),
        "group": train_group,
        "verbose": 100,
    }
    if valid_df is not None and len(valid_df) > 0:
        valid_sorted, valid_group = sort_for_ranker(valid_df, features)
        fit_kwargs["eval_set"] = [
            (
                valid_sorted.loc[:, list(features)],
                valid_sorted[TARGET_COL].to_numpy(dtype="float32"),
            )
        ]
        fit_kwargs["eval_group"] = [valid_group]

    LOGGER.info("Training XGBoost rank:ndcg ranker on %d rows", len(train_df))
    model.fit(**fit_kwargs)
    del train_sorted
    gc.collect()
    return model


def predict_xgb_ranker(
    model: object,
    df: pd.DataFrame,
    features: Sequence[str],
) -> np.ndarray:
    preds = model.predict(df.loc[:, list(features)])
    return np.asarray(preds, dtype="float32")


def catboost_feature_columns(
    base_features: Sequence[str],
    df: pd.DataFrame,
) -> Tuple[List[str], List[str]]:
    features = [col for col in base_features if col in df.columns]
    for col in CATBOOST_RAW_CATEGORICAL_COLS:
        if col in df.columns and col not in features:
            features.append(col)
    cat_features = [
        col
        for col in features
        if col in CATBOOST_RAW_CATEGORICAL_COLS
        or col in LOW_CARD_CATEGORICAL_COLS
        or col.endswith("_bucket")
    ]
    return features, cat_features


def prepare_catboost_frame(
    df: pd.DataFrame,
    features: Sequence[str],
    cat_features: Sequence[str],
) -> pd.DataFrame:
    frame = df.loc[:, list(features)].copy()
    for col in cat_features:
        if col in frame.columns:
            frame[col] = (
                frame[col].fillna(-1).astype("int64", errors="ignore").astype(str)
            )
    return frame


def fit_catboost_classifier(
    train_df: pd.DataFrame,
    valid_df: Optional[pd.DataFrame],
    target_col: str,
    features: Sequence[str],
    config: Config,
) -> Tuple[object, List[str], List[str]]:
    from catboost import CatBoostClassifier, Pool

    cb_features, cat_features = catboost_feature_columns(features, train_df)
    X_train = prepare_catboost_frame(train_df, cb_features, cat_features)
    train_pool = Pool(
        X_train,
        train_df[target_col].to_numpy(dtype="int8"),
        cat_features=cat_features,
    )
    eval_set = None
    if valid_df is not None and len(valid_df) > 0:
        X_valid = prepare_catboost_frame(valid_df, cb_features, cat_features)
        eval_set = Pool(
            X_valid,
            valid_df[target_col].to_numpy(dtype="int8"),
            cat_features=cat_features,
        )

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=min(config.n_estimators, 2000),
        learning_rate=config.learning_rate,
        depth=config.max_depth if 0 < config.max_depth <= 10 else 8,
        l2_leaf_reg=max(config.reg_lambda, 1.0),
        random_seed=config.seed + 41,
        thread_count=config.n_jobs,
        verbose=100,
        allow_writing_files=False,
    )
    LOGGER.info(
        "Training CatBoost classifier target=%s on %d rows",
        target_col,
        len(train_df),
    )
    model.fit(train_pool, eval_set=eval_set, use_best_model=eval_set is not None)
    return model, cb_features, cat_features


def fit_catboost_ranker(
    train_df: pd.DataFrame,
    valid_df: Optional[pd.DataFrame],
    features: Sequence[str],
    config: Config,
) -> Tuple[object, List[str], List[str]]:
    from catboost import CatBoostRanker, Pool

    cb_features, cat_features = catboost_feature_columns(features, train_df)
    train_sorted = train_df.sort_values(GROUP_COL, kind="mergesort")
    X_train = prepare_catboost_frame(train_sorted, cb_features, cat_features)
    train_pool = Pool(
        X_train,
        train_sorted[TARGET_COL].to_numpy(dtype="float32"),
        group_id=train_sorted[GROUP_COL],
        cat_features=cat_features,
    )
    eval_set = None
    if valid_df is not None and len(valid_df) > 0:
        valid_sorted = valid_df.sort_values(GROUP_COL, kind="mergesort")
        X_valid = prepare_catboost_frame(valid_sorted, cb_features, cat_features)
        eval_set = Pool(
            X_valid,
            valid_sorted[TARGET_COL].to_numpy(dtype="float32"),
            group_id=valid_sorted[GROUP_COL],
            cat_features=cat_features,
        )

    model = CatBoostRanker(
        loss_function="YetiRank",
        eval_metric="NDCG:top=5",
        iterations=min(config.n_estimators, 2000),
        learning_rate=config.learning_rate,
        depth=config.max_depth if 0 < config.max_depth <= 10 else 8,
        l2_leaf_reg=max(config.reg_lambda, 1.0),
        random_seed=config.seed + 43,
        thread_count=config.n_jobs,
        verbose=100,
        allow_writing_files=False,
    )
    LOGGER.info("Training CatBoost ranker on %d rows", len(train_df))
    model.fit(train_pool, eval_set=eval_set, use_best_model=eval_set is not None)
    return model, cb_features, cat_features


def predict_catboost(
    model: object,
    df: pd.DataFrame,
    features: Sequence[str],
    cat_features: Sequence[str],
) -> np.ndarray:
    X = prepare_catboost_frame(df, features, cat_features)
    if hasattr(model, "predict_proba"):
        preds = model.predict_proba(X)[:, 1]
    else:
        preds = model.predict(X)
    return np.asarray(preds, dtype="float32")


def fit_predict_base_model(
    model_name: OOFModelName,
    train_df: pd.DataFrame,
    valid_df: Optional[pd.DataFrame],
    apply_df: pd.DataFrame,
    features: Sequence[str],
    categorical_features: Sequence[str],
    config: Config,
    n_estimators: Optional[int] = None,
) -> Tuple[object, np.ndarray, Mapping[str, object]]:
    if model_name == "lgbm_ranker":
        assert_lgbm_feature_policy(features, categorical_features)
        model = fit_ranker(
            train_df,
            valid_df,
            features,
            categorical_features,
            config,
            n_estimators=n_estimators,
        )
        return model, predict_ranker(model, apply_df, features), {}
    if model_name == "lgbm_booking":
        assert_lgbm_feature_policy(features, categorical_features)
        model = fit_classifier(
            train_df,
            valid_df,
            BOOKING_COL,
            features,
            categorical_features,
            config,
            random_state_offset=11,
            n_estimators=n_estimators,
        )
        return model, predict_classifier(model, apply_df, features), {}
    if model_name == "lgbm_click":
        assert_lgbm_feature_policy(features, categorical_features)
        model = fit_classifier(
            train_df,
            valid_df,
            CLICK_COL,
            features,
            categorical_features,
            config,
            random_state_offset=17,
            n_estimators=n_estimators,
        )
        return model, predict_classifier(model, apply_df, features), {}
    if model_name == "xgb_ranker":
        model = fit_xgb_ranker(
            train_df,
            valid_df,
            features,
            config,
            n_estimators=n_estimators,
        )
        return model, predict_xgb_ranker(model, apply_df, features), {}
    if model_name == "catboost_booking":
        model, cb_features, cat_features = fit_catboost_classifier(
            train_df,
            valid_df,
            BOOKING_COL,
            features,
            config,
        )
        return (
            model,
            predict_catboost(model, apply_df, cb_features, cat_features),
            {"features": cb_features, "cat_features": cat_features},
        )
    if model_name == "catboost_ranker":
        model, cb_features, cat_features = fit_catboost_ranker(
            train_df,
            valid_df,
            features,
            config,
        )
        return (
            model,
            predict_catboost(model, apply_df, cb_features, cat_features),
            {"features": cb_features, "cat_features": cat_features},
        )
    raise ValueError(f"Unknown OOF model: {model_name}")


def config_payload(config: Config) -> Dict[str, object]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    }


def run_oof_experiment(config: Config) -> None:
    timer = timer_Factory(config.profile)
    model_names = available_oof_models(config)
    score_cols = [score_col_for_model(model_name) for model_name in model_names]
    LOGGER.info("Running OOF workflow with models: %s", model_names)

    with timer("oof: load training data"):
        train = load_dataset(config.train_path, config.sample_groups, config.seed)
    with timer("oof: add relevance labels"):
        add_relevance(train)

    fold_info: Dict[str, object] = {"strategy": "skipped"}
    oof_df: Optional[pd.DataFrame] = None
    meta_df: Optional[pd.DataFrame] = None
    metrics: Dict[str, object] = {}
    ensemble_weights: Tuple[float, ...] = tuple(
        float(1.0 / len(score_cols)) for _ in score_cols
    )
    final_method = "rank_average"
    final_oof_score = 0.0
    meta_feature_cols: List[str] = []

    if not config.skip_validation:
        with timer("oof: make grouped folds"):
            folds, fold_info = make_stratified_group_folds(
                train,
                config.oof_folds,
                config.seed,
            )
            LOGGER.info("OOF fold summary: %s", fold_info)

        oof_scores = {
            score_col: np.full(len(train), np.nan, dtype="float32")
            for score_col in score_cols
        }
        fold_metrics: List[Dict[str, object]] = []
        meta_frames: List[pd.DataFrame] = []
        feature_counts: List[int] = []

        for fold in range(int(np.max(folds)) + 1):
            LOGGER.info("Starting OOF fold %d", fold)
            fit_mask = folds != fold
            valid_mask = folds == fold
            train_part = train.loc[fit_mask].copy()
            valid_part = train.loc[valid_mask].copy()

            with timer(f"oof fold {fold}: build ranking features"):
                features, categorical_features = build_ranking_features(
                    train_part,
                    valid_part,
                    config,
                )
            with timer(f"oof fold {fold}: auxiliary position model feature"):
                features, _ = add_auxiliary_position_feature(
                    train_part,
                    valid_part,
                    features,
                    categorical_features,
                    config,
                )
            assert_lgbm_feature_policy(features, categorical_features)
            feature_counts.append(len(features))

            fold_scores: Dict[str, float] = {}
            for model_name, score_col in zip(model_names, score_cols):
                with timer(f"oof fold {fold}: train {model_name}"):
                    model, preds, _ = fit_predict_base_model(
                        model_name,
                        train_part,
                        valid_part,
                        valid_part,
                        features,
                        categorical_features,
                        config,
                    )
                valid_part[score_col] = preds
                oof_scores[score_col][valid_mask] = preds
                fold_scores[score_col] = ndcg_at_k(valid_part, score_col)
                del model
                gc.collect()

            fold_meta_cols = meta_candidate_columns(valid_part, score_cols, features)
            fold_meta = valid_part[
                [GROUP_COL, ITEM_COL, TARGET_COL, *fold_meta_cols]
            ].copy()
            fold_meta["_oof_fold"] = np.int16(fold)
            meta_frames.append(fold_meta)

            fold_metrics.append(
                {
                    "fold": fold,
                    "rows": int(valid_mask.sum()),
                    "groups": int(train.loc[valid_mask, GROUP_COL].nunique()),
                    "feature_count": len(features),
                    "scores": fold_scores,
                }
            )
            del train_part, valid_part, fold_meta
            gc.collect()

        oof_df = train[
            [
                GROUP_COL,
                ITEM_COL,
                TARGET_COL,
                BOOKING_COL,
                CLICK_COL,
                *[col for col in ("random_bool", "date_time") if col in train.columns],
            ]
        ].copy()
        for score_col in score_cols:
            if np.isnan(oof_scores[score_col]).any():
                raise ValueError(f"OOF score column {score_col} contains missing values.")
            oof_df[score_col] = oof_scores[score_col]

        with timer("oof: tune rank-average weights"):
            ensemble_weights, rankavg_score, rankavg_preds = tune_rank_average_weights(
                oof_df,
                score_cols,
                config.ensemble_weight_step,
            )
            oof_df["ensemble_rankavg_score"] = rankavg_preds
            final_oof_score = rankavg_score

        meta_df = pd.concat(meta_frames, axis=0).sort_index()
        meta_feature_cols = sorted(
            set().union(
                *[
                    set(frame.columns)
                    - {GROUP_COL, ITEM_COL, TARGET_COL, "_oof_fold"}
                    for frame in meta_frames
                ]
            )
        )
        align_meta_features(meta_df, meta_feature_cols)

        meta_score: Optional[float] = None
        if config.use_meta_ranker and len(score_cols) > 1:
            with timer("oof: fit cross-fitted meta-ranker"):
                meta_preds, meta_score = fit_meta_ranker_oof(
                    meta_df,
                    meta_feature_cols,
                    config,
                )
            oof_df.loc[meta_df.index, "meta_ranker_score"] = meta_preds
            if meta_score >= rankavg_score + config.stacking_min_delta:
                final_method = "meta_ranker"
                final_oof_score = meta_score
            LOGGER.info(
                "OOF meta-ranker ndcg_at_5=%.9f; rank-average ndcg_at_5=%.9f; "
                "selected=%s",
                meta_score,
                rankavg_score,
                final_method == "meta_ranker",
            )

        selected_score_cols = [*score_cols, "ensemble_rankavg_score"]
        if "meta_ranker_score" in oof_df.columns:
            selected_score_cols.append("meta_ranker_score")
        metrics = {
            "folding": fold_info,
            "base_models": model_names,
            "fold_metrics": fold_metrics,
            "feature_count_mean": float(np.mean(feature_counts)),
            "feature_count_min": int(np.min(feature_counts)),
            "feature_count_max": int(np.max(feature_counts)),
            "oof_scores": {
                score_col: ndcg_at_k(oof_df, score_col)
                for score_col in selected_score_cols
            },
            "segment_scores": segment_scores(oof_df, selected_score_cols),
            "ensemble_rankavg_weights": dict(zip(score_cols, ensemble_weights)),
            "ensemble_rankavg_weight_step": config.ensemble_weight_step,
            "final_method": final_method,
            "selected_oof_ndcg_at_5": final_oof_score,
            "stacking_min_delta": config.stacking_min_delta,
        }

        if config.validation_predictions_path is not None:
            config.validation_predictions_path.parent.mkdir(parents=True, exist_ok=True)
            oof_df.to_csv(config.validation_predictions_path, index=False)
            LOGGER.info(
                "Wrote OOF validation predictions to %s",
                config.validation_predictions_path,
            )

    if not config.no_final:
        with timer("final: load test data"):
            test = load_dataset(config.test_path, config.sample_groups, config.seed + 1)
        with timer("final: validate columns"):
            validate_train_test_columns(train, test)
        with timer("final: build ranking features"):
            features, categorical_features = build_ranking_features(train, test, config)
        with timer("final: auxiliary position model feature"):
            features, position_model = add_auxiliary_position_feature(
                train,
                test,
                features,
                categorical_features,
                config,
            )
        assert_lgbm_feature_policy(features, categorical_features)

        test_predictions = test[[GROUP_COL, ITEM_COL]].copy()
        final_models: Dict[str, object] = {}
        sidecars: Dict[str, Mapping[str, object]] = {}
        for model_name, score_col in zip(model_names, score_cols):
            with timer(f"final: train {model_name}"):
                model, preds, sidecar = fit_predict_base_model(
                    model_name,
                    train,
                    valid_df=None,
                    apply_df=test,
                    features=features,
                    categorical_features=categorical_features,
                    config=config,
                )
            test_predictions[score_col] = preds
            test[score_col] = preds
            final_models[model_name] = model
            sidecars[model_name] = sidecar
            gc.collect()

        if final_method == "meta_ranker" and meta_df is not None:
            test_meta_cols = meta_candidate_columns(test, score_cols, features)
            test_meta = test[[GROUP_COL, ITEM_COL, *test_meta_cols]].copy()
            align_meta_features(test_meta, meta_feature_cols)
            train_meta = cast(pd.DataFrame, meta_df).copy()
            align_meta_features(train_meta, meta_feature_cols)
            with timer("final: fit meta-ranker"):
                meta_model, final_preds = fit_final_meta_ranker(
                    train_meta,
                    test_meta,
                    meta_feature_cols,
                    config,
                )
            final_models["meta_ranker"] = meta_model
            test_predictions["final_score"] = final_preds
        else:
            test_predictions["final_score"] = weighted_rank_average(
                test_predictions,
                score_cols,
                ensemble_weights,
            )

        with timer("final: write predictions and submission"):
            pred_path = predictions_path(config.output_path)
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            test_predictions.to_csv(pred_path, index=False)
            submission = test_predictions.sort_values(
                [GROUP_COL, "final_score"],
                ascending=[True, False],
                kind="mergesort",
            )[[GROUP_COL, ITEM_COL]]
            config.output_path.parent.mkdir(parents=True, exist_ok=True)
            submission.to_csv(config.output_path, index=False)
            LOGGER.info("Wrote final test predictions to %s", pred_path)
            LOGGER.info("Wrote submission to %s", config.output_path)

        with timer("final: write model artifacts"):
            save_pickle(config.model_dir / "features_final.pkl", features)
            save_pickle(
                config.model_dir / "categorical_features_final.pkl",
                categorical_features,
            )
            if position_model is not None:
                save_pickle(config.model_dir / "position_final_model.pkl", position_model)
            for model_name, model in final_models.items():
                save_pickle(config.model_dir / f"{model_name}_final_model.pkl", model)
            if sidecars:
                save_pickle(config.model_dir / "oof_model_sidecars.pkl", sidecars)

        metrics["final_predictions_path"] = str(predictions_path(config.output_path))
        metrics["submission_path"] = str(config.output_path)

    if config.metrics_path is not None:
        write_json(
            config.metrics_path,
            {
                "config": config_payload(config),
                "metrics": metrics,
            },
        )
        LOGGER.info("Wrote metrics to %s", config.metrics_path)
