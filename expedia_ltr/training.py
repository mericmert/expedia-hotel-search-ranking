from __future__ import annotations

import gc
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd

from .artifacts import load_pickle, save_pickle, write_json
from .config import Config, GROUP_COL, ITEM_COL, LOGGER, TARGET_COL
from .data import add_relevance, load_dataset, validate_train_test_columns
from .features import build_ranking_features, make_group_folds
from .model import (
    POSITION_MODEL_FEATURE,
    fit_position_regressor,
    fit_ranker,
    predict_position_regressor,
    predict_ranker,
)
from .utils import timer_Factory, validation_audit_summary_path
from .validation import (
    make_group_split,
    ndcg_at_k,
    config_for_validation_split,
)


def add_auxiliary_position_feature(
    train_df: pd.DataFrame,
    apply_df: pd.DataFrame,
    features: List[str],
    categorical_features: List[str],
    config: Config,
) -> Tuple[List[str], Optional[object]]:
    if not config.use_aux_position_model:
        LOGGER.info("Skipping auxiliary position model feature by config.")
        return features, None

    if config.aux_position_model_path is not None:
        LOGGER.info(
            "Loading auxiliary position model from %s",
            config.aux_position_model_path,
        )
        position_model = load_pickle(config.aux_position_model_path)
        train_df[POSITION_MODEL_FEATURE] = predict_position_regressor(
            position_model,
            train_df,
            features,
        )
        apply_df[POSITION_MODEL_FEATURE] = predict_position_regressor(
            position_model,
            apply_df,
            features,
        )
        return [*features, POSITION_MODEL_FEATURE], position_model

    if "position" not in train_df.columns:
        LOGGER.warning("Skipping auxiliary position model; position is missing.")
        return features, None

    LOGGER.info("Adding auxiliary OOF position model feature")
    valid_position = train_df["position"].notna().to_numpy()
    if not valid_position.any():
        LOGGER.warning("Skipping auxiliary position model; no position values found.")
        return features, None

    target_mean = np.float32(
        np.log1p(train_df.loc[valid_position, "position"].astype("float32")).mean()
    )
    oof_pred = np.full(len(train_df), target_mean, dtype="float32")
    folds = make_group_folds(
        train_df[GROUP_COL], config.target_encoding_folds, config.seed + 101
    )
    n_folds = int(folds.max()) + 1

    if n_folds < 2:
        LOGGER.warning(
            "Only one fold available; auxiliary position model uses the global prior for train rows."
        )
    else:
        for fold in range(n_folds):
            holdout_mask = folds == fold
            fit_mask = (folds != fold) & valid_position
            if not fit_mask.any():
                continue
            fold_model = fit_position_regressor(
                train_df.loc[fit_mask],
                features,
                categorical_features,
                config,
            )
            oof_pred[holdout_mask] = predict_position_regressor(
                fold_model,
                train_df.loc[holdout_mask],
                features,
            )
            del fold_model
            gc.collect()

    train_df[POSITION_MODEL_FEATURE] = oof_pred
    position_model = fit_position_regressor(
        train_df,
        features,
        categorical_features,
        config,
    )
    apply_df[POSITION_MODEL_FEATURE] = predict_position_regressor(
        position_model,
        apply_df,
        features,
    )

    return [*features, POSITION_MODEL_FEATURE], position_model


def run_validation(
    config: Config,
) -> Tuple[Optional[int], List[str], Dict[str, object], bool, Tuple[float, ...]]:
    timer = timer_Factory(config.profile)

    with timer("validation: load training data"):
        train = load_dataset(config.train_path, config.sample_groups, config.seed)
    with timer("validation: add relevance labels"):
        add_relevance(train)
    with timer("validation: group split"):
        train_part, valid_part = make_group_split(train, config)
    del train
    gc.collect()

    with timer("validation: build ranking features"):
        features, categorical_features = build_ranking_features(
            train_part, valid_part, config
        )
    with timer("validation: auxiliary position model feature"):
        features, position_model = add_auxiliary_position_feature(
            train_part, valid_part, features, categorical_features, config
        )
    with timer("validation: fit ranker"):
        ranker = fit_ranker(
            train_part, valid_part, features, categorical_features, config
        )
    with timer("validation: predict scores"):
        valid_part["ranker_score"] = predict_ranker(ranker, valid_part, features)

    with timer("validation: metrics"):
        rng = np.random.default_rng(config.seed)
        valid_part["random_score"] = rng.random(len(valid_part), dtype=np.float32)
        ranker_ndcg = ndcg_at_k(valid_part, "ranker_score")
        metrics: Dict[str, object] = {
            "random_ndcg_at_5": ndcg_at_k(valid_part, "random_score"),
            "ranker_ndcg_at_5": ranker_ndcg,
            "selected_ndcg_at_5": ranker_ndcg,
        }

    LOGGER.info("Validation metrics: %s", metrics)
    with timer("validation: write metrics and artifacts"):
        write_json(
            config.metrics_path,
            {
                "config": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in asdict(config).items()
                },
                "metrics": metrics,
                "feature_count": len(features),
                "features": features,
            },
        )

        if config.validation_predictions_path is not None:
            cols = [
                GROUP_COL,
                ITEM_COL,
                TARGET_COL,
                "ranker_score",
            ]
            config.validation_predictions_path.parent.mkdir(parents=True, exist_ok=True)
            valid_part[cols].to_csv(config.validation_predictions_path, index=False)
            LOGGER.info(
                "Wrote validation predictions to %s", config.validation_predictions_path
            )

        best_iteration = getattr(ranker, "best_iteration_", None)
        artifacts = {
            "ranker_validation_model.pkl": ranker,
            "position_validation_model.pkl": position_model,
            "features.pkl": features,
            "categorical_features.pkl": categorical_features,
        }
        for name, obj in artifacts.items():
            if obj is not None:
                artifact_name = (
                    name if name.endswith(".pkl") else f"{name}_validation_model.pkl"
                )
                save_pickle(config.model_dir / artifact_name, obj)

    del train_part, valid_part, ranker
    gc.collect()
    return (
        best_iteration,
        features,
        metrics,
        False,
        (),
    )


def run_multi_split_validation(
    config: Config,
) -> Tuple[Optional[int], List[str], bool, Tuple[float, ...]]:
    requested_splits = config.validation_splits
    if not requested_splits:
        (
            best_iteration,
            features,
            _,
            use_ranker_blend_for_final,
            selected_ranker_blend_weights,
        ) = run_validation(config)
        return (
            best_iteration,
            features,
            use_ranker_blend_for_final,
            selected_ranker_blend_weights,
        )

    strategies = tuple(dict.fromkeys((config.split_strategy, *requested_splits)))
    LOGGER.info("Running validation audit across splits: %s", strategies)
    primary_result: Optional[
        Tuple[Optional[int], List[str], bool, Tuple[float, ...]]
    ] = None
    summary: Dict[str, object] = {
        "primary_split_strategy": config.split_strategy,
        "split_strategies": strategies,
        "results": {},
    }

    for split_strategy in strategies:
        split_config = config_for_validation_split(config, split_strategy)
        (
            best_iteration,
            features,
            metrics,
            use_ranker_blend_for_final,
            selected_ranker_blend_weights,
        ) = run_validation(split_config)
        cast(Dict[str, object], summary["results"])[split_strategy] = {
            "metrics": metrics,
            "best_iteration": best_iteration,
            "feature_count": len(features),
            "metrics_path": str(split_config.metrics_path)
            if split_config.metrics_path is not None
            else None,
            "model_dir": str(split_config.model_dir),
        }
        if split_strategy == config.split_strategy or primary_result is None:
            primary_result = (
                best_iteration,
                features,
                use_ranker_blend_for_final,
                selected_ranker_blend_weights,
            )

    write_json(validation_audit_summary_path(config.metrics_path), summary)
    if primary_result is None:
        raise ValueError("No validation split results were produced.")
    return primary_result


def run_final(
    config: Config,
    best_iteration: Optional[int],
    use_ranker_blend_for_final: bool,
    ranker_blend_weights: Tuple[float, ...],
) -> None:
    timer = timer_Factory(config.profile)

    with timer("final: load training data"):
        train = load_dataset(config.train_path, config.sample_groups, config.seed)
    with timer("final: load test data"):
        test = load_dataset(config.test_path, config.sample_groups, config.seed + 1)
    with timer("final: validate columns and labels"):
        validate_train_test_columns(train, test)
        add_relevance(train)

    with timer("final: build ranking features"):
        features, categorical_features = build_ranking_features(train, test, config)
    with timer("final: auxiliary position model feature"):
        features, position_model = add_auxiliary_position_feature(
            train, test, features, categorical_features, config
        )
    final_estimators = (
        best_iteration if best_iteration and best_iteration > 0 else config.n_estimators
    )
    LOGGER.info("Final ranker n_estimators=%s", final_estimators)
    with timer("final: fit ranker"):
        ranker = fit_ranker(
            train,
            valid_df=None,
            features=features,
            categorical_features=categorical_features,
            config=config,
            n_estimators=final_estimators,
        )
    with timer("final: predict scores"):
        test["ranker_score"] = predict_ranker(ranker, test, features)
    if use_ranker_blend_for_final or ranker_blend_weights:
        LOGGER.warning("Ignoring removed same-feature ranker blend settings.")
    final_score_col = "ranker_score"

    with timer("final: write submission"):
        submission = test.sort_values(
            [GROUP_COL, final_score_col], ascending=[True, False], kind="mergesort"
        )[[GROUP_COL, ITEM_COL]]
        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(config.output_path, index=False)
        LOGGER.info(
            "Wrote submission with shape %s to %s",
            submission.shape,
            config.output_path,
        )

    with timer("final: write model artifacts"):
        save_pickle(config.model_dir / "ranker_final_model.pkl", ranker)
        if position_model is not None:
            save_pickle(config.model_dir / "position_final_model.pkl", position_model)
        save_pickle(config.model_dir / "features_final.pkl", features)
        save_pickle(
            config.model_dir / "categorical_features_final.pkl", categorical_features
        )
