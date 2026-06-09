from __future__ import annotations

import gc
from typing import Optional, Sequence, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import Config, GROUP_COL, LOGGER, TARGET_COL

POSITION_MODEL_FEATURE = "position_model_log1p_pred"


def make_lgbm_ranker(
    config: Config, n_estimators: Optional[int] = None
) -> lgb.LGBMRanker:
    return lgb.LGBMRanker(
        objective=config.rank_objective,
        metric="ndcg",
        boosting_type="gbdt",
        n_estimators=n_estimators or config.n_estimators,
        learning_rate=config.learning_rate,
        num_leaves=config.num_leaves,
        max_depth=config.max_depth,
        min_child_samples=config.min_child_samples,
        subsample=config.subsample,
        subsample_freq=1,
        colsample_bytree=config.colsample_bytree,
        reg_alpha=config.reg_alpha,
        reg_lambda=config.reg_lambda,
        random_state=config.seed,
        n_jobs=config.n_jobs,
        label_gain=[0, 1, 3, 7, 15, 31],
        lambdarank_truncation_level=config.lambdarank_truncation_level,
        importance_type="gain",
        force_col_wise=True,
    )


def make_lgbm_position_regressor(config: Config) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        boosting_type="gbdt",
        n_estimators=min(config.n_estimators, 200),
        learning_rate=config.learning_rate,
        num_leaves=min(config.num_leaves, 64),
        max_depth=config.max_depth,
        min_child_samples=config.min_child_samples,
        subsample=config.subsample,
        subsample_freq=1,
        colsample_bytree=config.colsample_bytree,
        reg_alpha=config.reg_alpha,
        reg_lambda=config.reg_lambda,
        random_state=config.seed + 101,
        n_jobs=config.n_jobs,
        importance_type="gain",
        force_col_wise=True,
    )


def make_lgbm_classifier(
    config: Config,
    random_state_offset: int = 0,
    n_estimators: Optional[int] = None,
) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        metric="binary_logloss",
        boosting_type="gbdt",
        n_estimators=n_estimators or config.n_estimators,
        learning_rate=config.learning_rate,
        num_leaves=config.num_leaves,
        max_depth=config.max_depth,
        min_child_samples=config.min_child_samples,
        subsample=config.subsample,
        subsample_freq=1,
        colsample_bytree=config.colsample_bytree,
        reg_alpha=config.reg_alpha,
        reg_lambda=config.reg_lambda,
        random_state=config.seed + random_state_offset,
        n_jobs=config.n_jobs,
        importance_type="gain",
        force_col_wise=True,
    )


def sort_for_ranker(
    df: pd.DataFrame, features: Sequence[str]
) -> Tuple[pd.DataFrame, np.ndarray]:
    sorted_df = df.sort_values(GROUP_COL, kind="mergesort").reset_index(drop=True)
    group = sorted_df.groupby(GROUP_COL, sort=False).size().to_numpy(dtype="int32")
    return sorted_df, group


def fit_ranker(
    train_df: pd.DataFrame,
    valid_df: Optional[pd.DataFrame],
    features: Sequence[str],
    categorical_features: Sequence[str],
    config: Config,
    n_estimators: Optional[int] = None,
) -> lgb.LGBMRanker:
    train_sorted, train_group = sort_for_ranker(train_df, features)
    X_train = train_sorted[list(features)]
    y_train = train_sorted[TARGET_COL].to_numpy(dtype="int8")
    model = make_lgbm_ranker(config, n_estimators=n_estimators)

    fit_kwargs = {
        "X": X_train,
        "y": y_train,
        "group": train_group,
        "categorical_feature": list(categorical_features),
    }

    callbacks = []
    if valid_df is not None and len(valid_df) > 0:
        valid_sorted, valid_group = sort_for_ranker(valid_df, features)
        fit_kwargs["eval_set"] = [
            (
                valid_sorted[list(features)],
                valid_sorted[TARGET_COL].to_numpy(dtype="int8"),
            )
        ]
        fit_kwargs["eval_group"] = [valid_group]
        fit_kwargs["eval_at"] = [5]
        callbacks = [
            lgb.early_stopping(config.early_stopping_rounds, verbose=True),
            lgb.log_evaluation(period=100),
        ]
        fit_kwargs["callbacks"] = callbacks

    LOGGER.info(
        "Training LightGBM ranker objective=%s on %d rows",
        config.rank_objective,
        len(train_sorted),
    )
    model.fit(**fit_kwargs)
    best_iter = getattr(model, "best_iteration_", None)
    if best_iter:
        LOGGER.info("Ranker best_iteration=%s", best_iter)

    del train_sorted, X_train, y_train
    gc.collect()
    return model


def fit_position_regressor(
    train_df: pd.DataFrame,
    features: Sequence[str],
    categorical_features: Sequence[str],
    config: Config,
) -> lgb.LGBMRegressor:
    if "position" not in train_df.columns:
        raise ValueError("position is required to fit the auxiliary position model.")

    fit_mask = train_df["position"].notna()
    fit_df = train_df.loc[fit_mask]
    X_train = fit_df.loc[:, list(features)]
    y_train = np.log1p(fit_df["position"].astype("float32")).to_numpy(dtype="float32")
    model = make_lgbm_position_regressor(config)

    LOGGER.info("Training auxiliary position regressor on %d rows", len(fit_df))
    model.fit(
        X=X_train,
        y=y_train,
        categorical_feature=list(categorical_features),
    )
    del fit_df, X_train, y_train
    gc.collect()
    return model


def fit_classifier(
    train_df: pd.DataFrame,
    valid_df: Optional[pd.DataFrame],
    target_col: str,
    features: Sequence[str],
    categorical_features: Sequence[str],
    config: Config,
    random_state_offset: int = 0,
    n_estimators: Optional[int] = None,
) -> lgb.LGBMClassifier:
    X_train = train_df.loc[:, list(features)]
    y_train = train_df[target_col].to_numpy(dtype="int8")
    model = make_lgbm_classifier(
        config,
        random_state_offset=random_state_offset,
        n_estimators=n_estimators,
    )

    fit_kwargs = {
        "X": X_train,
        "y": y_train,
        "categorical_feature": list(categorical_features),
    }
    if valid_df is not None and len(valid_df) > 0:
        fit_kwargs["eval_set"] = [
            (
                valid_df.loc[:, list(features)],
                valid_df[target_col].to_numpy(dtype="int8"),
            )
        ]
        fit_kwargs["callbacks"] = [
            lgb.early_stopping(config.early_stopping_rounds, verbose=True),
            lgb.log_evaluation(period=100),
        ]

    LOGGER.info(
        "Training LightGBM classifier target=%s on %d rows",
        target_col,
        len(train_df),
    )
    model.fit(**fit_kwargs)
    best_iter = getattr(model, "best_iteration_", None)
    if best_iter:
        LOGGER.info("Classifier target=%s best_iteration=%s", target_col, best_iter)

    del X_train, y_train
    gc.collect()
    return model


def predict_position_regressor(
    model: lgb.LGBMRegressor, df: pd.DataFrame, features: Sequence[str]
) -> np.ndarray:
    preds = model.predict(df.loc[:, list(features)])
    return np.asarray(preds, dtype=np.float32)


def predict_ranker(
    model: lgb.LGBMRanker, df: pd.DataFrame, features: Sequence[str]
) -> np.ndarray:
    best_iter = getattr(model, "best_iteration_", None)
    preds = model.predict(
        df.loc[:, list(features)],
        num_iteration=best_iter,
    )
    return np.asarray(preds, dtype=np.float32)


def predict_classifier(
    model: lgb.LGBMClassifier, df: pd.DataFrame, features: Sequence[str]
) -> np.ndarray:
    best_iter = getattr(model, "best_iteration_", None)
    preds = model.predict_proba(
        df.loc[:, list(features)],
        num_iteration=best_iter,
    )[:, 1]
    return np.asarray(preds, dtype=np.float32)
