from __future__ import annotations

import math
import unittest
from pathlib import Path

import pandas as pd

from expedia_ltr.config import Config, validate_config
from expedia_ltr.features import (
    add_oof_position_aggregates,
    add_oof_target_encodings,
    make_group_folds,
)
from expedia_ltr.validation import make_stratified_group_folds, ndcg_at_k


def minimal_config(**overrides) -> Config:
    values = {
        "train_path": Path("train.parquet"),
        "test_path": Path("test.parquet"),
        "output_path": Path("submission.csv"),
        "model_dir": Path("models"),
        "validation_predictions_path": Path("validation_predictions.csv"),
        "metrics_path": Path("metrics.json"),
        "split_strategy": "time",
        "validation_splits": (),
        "valid_fraction": 0.15,
        "workflow": "oof",
        "oof_folds": 5,
        "oof_models": ("lgbm_ranker",),
        "ensemble_weight_step": 0.05,
        "use_meta_ranker": True,
        "stacking_min_delta": 0.003,
        "seed": 7,
        "target_encoding_folds": 2,
        "target_encoding_smoothing": 40.0,
        "n_estimators": 10,
        "learning_rate": 0.05,
        "num_leaves": 15,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "lambdarank_truncation_level": 5,
        "early_stopping_rounds": 5,
        "rank_objective": "lambdarank",
        "ranker_blend_seeds": (),
        "ranker_blend_objectives": (),
        "ranker_blend_weight_step": 0.05,
        "ranker_blend_mode": "raw",
        "n_jobs": 1,
        "sample_groups": 0,
        "skip_validation": False,
        "no_final": True,
        "use_raw_prop_id": False,
        "use_aux_position_model": False,
        "aux_position_model_path": None,
        "profile": False,
    }
    values.update(overrides)
    return Config(**values)


class ValidationAndFeatureTests(unittest.TestCase):
    def test_ndcg_at_5_uses_exponential_gain(self) -> None:
        df = pd.DataFrame(
            {
                "srch_id": [1, 1, 1],
                "relevance": [5, 1, 0],
                "score": [0.1, 0.3, 0.2],
            }
        )
        actual = ndcg_at_k(df, "score", k=5)
        dcg = 1.0 / math.log2(2) + 31.0 / math.log2(4)
        idcg = 31.0 / math.log2(2) + 1.0 / math.log2(3)
        self.assertAlmostEqual(actual, dcg / idcg)

    def test_stratified_group_folds_keep_searches_intact(self) -> None:
        df = pd.DataFrame(
            {
                "srch_id": [1, 1, 2, 2, 3, 3, 4, 4],
                "random_bool": [0, 0, 0, 0, 1, 1, 1, 1],
                "booking_bool": [1, 0, 0, 0, 1, 0, 0, 0],
                "date_time": pd.to_datetime(
                    [
                        "2013-01-01",
                        "2013-01-01",
                        "2013-01-02",
                        "2013-01-02",
                        "2013-02-01",
                        "2013-02-01",
                        "2013-02-02",
                        "2013-02-02",
                    ]
                ),
            }
        )
        folds, info = make_stratified_group_folds(df, n_splits=2, seed=3)
        self.assertEqual(info["n_splits"], 2)
        for _, group_folds in pd.Series(folds).groupby(df["srch_id"]):
            self.assertEqual(group_folds.nunique(), 1)

    def test_oof_target_encoding_does_not_see_own_unique_key(self) -> None:
        df = pd.DataFrame(
            {
                "srch_id": [1, 2, 3, 4],
                "prop_id": [101, 102, 103, 104],
                "booking_bool": [1, 0, 0, 0],
                "click_bool": [1, 1, 0, 0],
                "relevance": [5, 1, 0, 0],
                "dcg_gain": [31, 1, 0, 0],
            }
        )
        config = minimal_config(seed=11, target_encoding_folds=2)
        folds = make_group_folds(df["srch_id"], 2, 11)
        add_oof_target_encodings(df, config)

        for row_idx, fold in enumerate(folds):
            fit_prior = df.loc[folds != fold, "relevance"].mean()
            self.assertAlmostEqual(df.loc[row_idx, "prop_relevance_te"], fit_prior)

    def test_oof_position_aggregate_does_not_see_own_unique_key(self) -> None:
        df = pd.DataFrame(
            {
                "srch_id": [1, 2, 3, 4],
                "prop_id": [101, 102, 103, 104],
                "position": [1, 2, 3, 4],
                "random_bool": [0, 0, 0, 0],
            }
        )
        config = minimal_config(seed=17, target_encoding_folds=2)
        folds = make_group_folds(df["srch_id"], 2, 17)
        add_oof_position_aggregates(df, config)

        for row_idx, fold in enumerate(folds):
            fit_prior = df.loc[folds != fold, "position"].mean()
            self.assertAlmostEqual(df.loc[row_idx, "prop_position_mean"], fit_prior)

    def test_oof_config_rejects_raw_prop_id_for_lightgbm(self) -> None:
        config = minimal_config(use_raw_prop_id=True)
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_saved_blend_regression_stays_near_known_score(self) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / "validation_predictions_lambdarank_xendcg_blend.csv"
        )
        if not path.exists():
            self.skipTest("saved blend validation predictions are not present")
        df = pd.read_csv(path)
        score_col = "ranker_blend_score" if "ranker_blend_score" in df else "ranker_score"
        score = ndcg_at_k(df, score_col)
        self.assertGreater(score, 0.426)
        self.assertLess(score, 0.429)


if __name__ == "__main__":
    unittest.main()
