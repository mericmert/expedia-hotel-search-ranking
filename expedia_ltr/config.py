from __future__ import annotations

import argparse
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Optional, Sequence, Tuple, cast

from pandas.errors import PerformanceWarning

LOGGER = logging.getLogger("expedia_ltr")


GROUP_COL = "srch_id"
ITEM_COL = "prop_id"
TARGET_COL = "relevance"
GAIN_COL = "dcg_gain"
BOOKING_COL = "booking_bool"
CLICK_COL = "click_bool"

VALID_SPLIT_STRATEGIES = ("time", "random", "srch_id_quantile")
VALID_RANKER_BLEND_MODES = ("raw", "rank_pct")
VALID_RANK_OBJECTIVES = ("lambdarank", "rank_xendcg")
VALID_WORKFLOWS = ("oof", "legacy")
VALID_OOF_MODELS = (
    "lgbm_ranker",
    "lgbm_booking",
    "lgbm_click",
    "xgb_ranker",
    "catboost_booking",
    "catboost_ranker",
)

SplitStrategy = Literal["time", "random", "srch_id_quantile"]
RankerBlendMode = Literal["raw", "rank_pct"]
RankObjective = Literal["lambdarank", "rank_xendcg"]
Workflow = Literal["oof", "legacy"]
OOFModelName = Literal[
    "lgbm_ranker",
    "lgbm_booking",
    "lgbm_click",
    "xgb_ranker",
    "catboost_booking",
    "catboost_ranker",
]

TRAIN_ONLY_COLS = frozenset(
    {
        "position",
        "gross_bookings_usd",
        CLICK_COL,
        BOOKING_COL,
        TARGET_COL,
        GAIN_COL,
    }
)

MISSINGNESS_EXCLUDE_COLS = frozenset(TRAIN_ONLY_COLS | {GROUP_COL})

RELATIVE_FEATURE_BASES = (
    "price_usd",
    "price_log1p",  # to reduce the impact of very large prices.
    "price_per_night",  # price_usd / srch_length_of_stay
    "price_per_room_night",  # price_usd / (srch_length_of_stay * srch_room_count)
    "price_per_person",  # price_usd / (srch_adults_count + srch_children_count)
    "prop_starrating",
    "prop_review_score",
    "prop_location_score1",
    "prop_location_score2",
    "prop_log_historical_price",
    "orig_destination_distance",
    "srch_query_affinity_score",
    "value_score",  # (star + review + loc1 + loc2) / (price_log1p + 1.0)
    "comp_rate_missing_count",
    "comp_rate_observed_count",
    "comp_rate_advantage_count",
    "comp_rate_disadvantage_count",
    "comp_rate_same_count",
    "comp_rate_mean",
    "comp_inv_missing_count",
    "comp_inv_observed_count",
    "comp_inv_advantage_count",
    "comp_inv_disadvantage_count",
    "comp_inv_mean",
    "comp_pct_missing_count",
    "comp_pct_observed_count",
    "comp_pct_mean",
    "comp_pct_min",
    "comp_pct_max",
    "comp_total_missing_count",
)


LOW_CARD_CATEGORICAL_COLS = (
    "site_id",
    "visitor_location_country_id",
    "prop_country_id",
    "prop_brand_bool",
    "promotion_flag",
    "random_bool",
    "srch_saturday_night_bool",
    "search_month",
    "search_dayofweek",
    "search_hour",
    "search_weekofyear",
    "checkin_month",
    "checkin_dayofweek",
    "family_trip_bool",  # True when children are included or traveler count is large.
    "business_trip_bool",  # True for short, non-family, non-Saturday trips
    "international_bool",  # custom feature based on prop_country_id != visitor_location_country_id
    "early_night",  # True when the search happened late evening or after midnight.
)

TARGET_ENCODING_TARGETS: Tuple[Tuple[str, str], ...] = (
    (TARGET_COL, "relevance"),
    (GAIN_COL, "gain"),
    (BOOKING_COL, "booking"),
    (CLICK_COL, "click"),
)

TARGET_KEY_SPECS: Tuple[Tuple[str, str], ...] = (
    ("prop_id", "prop"),
    ("srch_destination_id", "dest"),
    ("prop_country_id", "prop_country"),
    ("site_id", "site"),
    ("visitor_location_country_id", "visitor_country"),
    ("_key_prop_dest", "prop_dest"),
    ("_key_prop_country_visitor_country", "prop_country_visitor_country"),
    ("_key_dest_site", "dest_site"),
    ("price_usd_bucket", "price_bucket"),
    ("price_per_night_bucket", "price_per_night_bucket"),
    ("price_per_room_night_bucket", "price_room_night_bucket"),
    ("prop_log_historical_price_bucket", "hist_price_bucket"),
    ("srch_booking_window_bucket", "booking_window_bucket"),
    ("srch_length_of_stay_bucket", "stay_bucket"),
    ("_key_dest_price_bucket", "dest_price_bucket"),
    ("_key_site_price_bucket", "site_price_bucket"),
    ("_key_prop_country_price_bucket", "prop_country_price_bucket"),
    ("_key_dest_booking_window_bucket", "dest_booking_window_bucket"),
    ("_key_dest_stay_bucket", "dest_stay_bucket"),
)

POSITION_AGG_KEY_SPECS: Tuple[Tuple[str, str], ...] = (
    ("prop_id", "prop"),
    ("_key_prop_dest", "prop_dest"),
    ("_key_dest_site", "dest_site"),
    ("random_bool", "random"),
)

FREQUENCY_KEY_SPECS: Tuple[Tuple[str, str], ...] = (
    ("prop_id", "prop"),
    ("srch_destination_id", "dest"),
    ("prop_country_id", "prop_country"),
    ("site_id", "site"),
    ("visitor_location_country_id", "visitor_country"),
    ("_key_prop_dest", "prop_dest"),
    ("_key_dest_site", "dest_site"),
    ("price_usd_bucket", "price_bucket"),
    ("price_per_night_bucket", "price_per_night_bucket"),
    ("price_per_room_night_bucket", "price_room_night_bucket"),
    ("prop_log_historical_price_bucket", "hist_price_bucket"),
    ("srch_booking_window_bucket", "booking_window_bucket"),
    ("srch_length_of_stay_bucket", "stay_bucket"),
    ("_key_dest_price_bucket", "dest_price_bucket"),
    ("_key_site_price_bucket", "site_price_bucket"),
    ("_key_prop_country_price_bucket", "prop_country_price_bucket"),
    ("_key_dest_booking_window_bucket", "dest_booking_window_bucket"),
    ("_key_dest_stay_bucket", "dest_stay_bucket"),
)

AFFINITY_KEY_SPECS: Tuple[Tuple[str, str, str], ...] = (
    ("prop_dest_affinity", "prop_id", "srch_destination_id"),
    ("prop_site_affinity", "prop_id", "site_id"),
    ("prop_visitor_country_affinity", "prop_id", "visitor_location_country_id"),
    (
        "prop_country_visitor_country_affinity",
        "prop_country_id",
        "visitor_location_country_id",
    ),
    (
        "dest_visitor_country_affinity",
        "srch_destination_id",
        "visitor_location_country_id",
    ),
    ("dest_site_affinity", "srch_destination_id", "site_id"),
    ("prop_country_site_affinity", "prop_country_id", "site_id"),
)

BUCKET_SOURCE_SPECS: Tuple[Tuple[str, str, int], ...] = (
    ("price_usd", "price_usd_bucket", 20),
    ("price_per_night", "price_per_night_bucket", 20),
    ("price_per_room_night", "price_per_room_night_bucket", 16),
    ("prop_log_historical_price", "prop_log_historical_price_bucket", 16),
    ("srch_booking_window", "srch_booking_window_bucket", 16),
    ("srch_length_of_stay", "srch_length_of_stay_bucket", 12),
)

BUCKET_CROSS_KEY_SPECS: Mapping[str, Tuple[str, str]] = {
    "_key_dest_price_bucket": ("srch_destination_id", "price_usd_bucket"),
    "_key_site_price_bucket": ("site_id", "price_usd_bucket"),
    "_key_prop_country_price_bucket": ("prop_country_id", "price_usd_bucket"),
    "_key_dest_booking_window_bucket": (
        "srch_destination_id",
        "srch_booking_window_bucket",
    ),
    "_key_dest_stay_bucket": ("srch_destination_id", "srch_length_of_stay_bucket"),
}

SEGMENT_PROFILE_SPECS: Tuple[Tuple[str, str], ...] = (
    ("prop_country", "prop_country_id"),
    ("dest", "srch_destination_id"),
    ("site", "site_id"),
    ("visitor_country", "visitor_location_country_id"),
    ("dest_site", "_key_dest_site"),
)

SEGMENT_PROFILE_SOURCE_COLS = (
    "price_usd",
    "price_per_night",
    "price_per_room_night",
    "prop_starrating",
    "prop_review_score",
    "prop_location_score1",
    "prop_location_score2",
    "prop_log_historical_price",
    "promotion_flag",
)

MISSING_VALUE_FEATURE_COLS = (
    "visitor_hist_starrating",
    "visitor_hist_adr_usd",
    "prop_review_score",
    "prop_location_score2",
    "orig_destination_distance",
    "srch_query_affinity_score",
    "prop_log_historical_price",
)

ZERO_AS_MISSING_COLS = {"prop_log_historical_price"}

PROP_PROFILE_STATS = ("mean", "median")

PROP_PROFILE_SOURCE_COLS = (
    "price_usd",
    "prop_starrating",
    "prop_review_score",
    "prop_location_score1",
    "prop_location_score2",
    "prop_log_historical_price",
    "srch_query_affinity_score",
    "orig_destination_distance",
    "promotion_flag",
)

PROP_PROFILE_DELTA_COLS = (
    "price_usd",
    "prop_starrating",
    "prop_review_score",
    "prop_location_score1",
    "prop_location_score2",
    "prop_log_historical_price",
    "srch_query_affinity_score",
    "orig_destination_distance",
)

HELPER_PREFIXES = ("_key_",)


@dataclass(frozen=True, slots=True)
class Config:
    train_path: Path
    test_path: Path
    output_path: Path
    model_dir: Path
    validation_predictions_path: Optional[Path]
    metrics_path: Optional[Path]
    split_strategy: SplitStrategy
    validation_splits: Tuple[SplitStrategy, ...]
    valid_fraction: float
    workflow: Workflow
    oof_folds: int
    oof_models: Tuple[OOFModelName, ...]
    ensemble_weight_step: float
    use_meta_ranker: bool
    stacking_min_delta: float
    seed: int
    target_encoding_folds: int
    target_encoding_smoothing: float
    n_estimators: int
    learning_rate: float
    num_leaves: int
    max_depth: int
    min_child_samples: int
    subsample: float
    colsample_bytree: float
    reg_alpha: float
    reg_lambda: float
    lambdarank_truncation_level: int
    early_stopping_rounds: int
    rank_objective: RankObjective
    ranker_blend_seeds: Tuple[int, ...]
    ranker_blend_objectives: Tuple[RankObjective, ...]
    ranker_blend_weight_step: float
    ranker_blend_mode: RankerBlendMode
    n_jobs: int
    sample_groups: int
    skip_validation: bool
    no_final: bool
    use_raw_prop_id: bool
    use_aux_position_model: bool
    aux_position_model_path: Optional[Path]
    profile: bool


def validate_config(config: Config) -> None:
    if not 0.0 <= config.valid_fraction <= 0.5:
        raise ValueError("valid_fraction must be in the range [0.0, 0.5].")

    if config.target_encoding_folds < 2:
        raise ValueError("target_encoding_folds must be at least 2.")

    if config.workflow == "oof":
        if config.oof_folds < 2:
            raise ValueError("--oof-folds must be at least 2.")
        if not config.oof_models:
            raise ValueError("--oof-models must select at least one base model.")
        if not 0.0 < config.ensemble_weight_step <= 1.0:
            raise ValueError("--ensemble-weight-step must be in (0, 1].")
        ensemble_weight_units = round(1.0 / config.ensemble_weight_step)
        if ensemble_weight_units > 100:
            raise ValueError("--ensemble-weight-step cannot be smaller than 0.01.")
        if config.stacking_min_delta < 0.0:
            raise ValueError("--stacking-min-delta must be non-negative.")
        if config.use_raw_prop_id:
            raise ValueError(
                "--use-raw-prop-id is not allowed in the OOF LightGBM workflow. "
                "Use the CatBoost models for raw high-cardinality IDs."
            )

    if config.target_encoding_smoothing < 0.0:
        raise ValueError("target_encoding_smoothing must be non-negative.")

    if config.n_estimators <= 0:
        raise ValueError("n_estimators must be positive.")

    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")

    if config.num_leaves <= 1:
        raise ValueError("num_leaves must be greater than 1.")

    if not 0.0 < config.subsample <= 1.0:
        raise ValueError("--subsample must be in (0, 1].")

    if not 0.0 < config.colsample_bytree <= 1.0:
        raise ValueError("--colsample-bytree must be in (0, 1].")

    if config.sample_groups < 0:
        raise ValueError("--sample-groups cannot be negative.")

    if config.ranker_blend_seeds or config.ranker_blend_objectives:
        raise ValueError(
            "Same-feature ranker blending has been removed. Use the OOF base-model "
            "ensemble instead."
        )

    if config.seed in config.ranker_blend_seeds:
        raise ValueError("--ranker-blend-seeds should not include the primary --seed.")

    if any(seed < 0 for seed in config.ranker_blend_seeds):
        raise ValueError("--ranker-blend-seeds must be non-negative integers.")

    if not 0.0 < config.ranker_blend_weight_step <= 1.0:
        raise ValueError("--ranker-blend-weight-step must be in (0, 1].")

    blend_weight_units = round(1.0 / config.ranker_blend_weight_step)
    if blend_weight_units > 100:
        raise ValueError("--ranker-blend-weight-step cannot be smaller than 0.01.")

    if (
        config.ranker_blend_seeds
        and config.ranker_blend_objectives
        and len(config.ranker_blend_objectives) not in (1, len(config.ranker_blend_seeds))
    ):
        raise ValueError(
            "--ranker-blend-objectives must contain either one objective to reuse "
            "for every blend seed or one objective per --ranker-blend-seeds value."
        )

    if config.aux_position_model_path is not None:
        if not config.use_aux_position_model:
            raise ValueError(
                "--aux-position-model-path cannot be used with "
                "--no-aux-position-model."
            )
        if not config.aux_position_model_path.exists():
            raise FileNotFoundError(config.aux_position_model_path)


def parse_validation_splits(value: str) -> Tuple[SplitStrategy, ...]:
    if not value.strip():
        return ()
    requested = tuple(part.strip() for part in value.split(",") if part.strip())
    invalid = sorted(set(requested) - set(VALID_SPLIT_STRATEGIES))
    if invalid:
        raise ValueError(
            f"Unknown validation split strategies: {invalid}. "
            f"Valid choices are: {VALID_SPLIT_STRATEGIES}"
        )
    # Remove duplicates
    unique_requested = tuple(dict.fromkeys(requested))
    return cast(Tuple[SplitStrategy, ...], unique_requested)


def parse_ranker_blend_seeds(value: str) -> Tuple[int, ...]:
    if not value.strip():
        return ()
    seeds = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    return tuple(dict.fromkeys(seeds))


def parse_rank_objectives(value: str) -> Tuple[RankObjective, ...]:
    if not value.strip():
        return ()
    requested = tuple(part.strip() for part in value.split(",") if part.strip())
    invalid = sorted(set(requested) - set(VALID_RANK_OBJECTIVES))
    if invalid:
        raise ValueError(
            f"Unknown rank objectives: {invalid}. "
            f"Valid choices are: {VALID_RANK_OBJECTIVES}"
        )
    return cast(Tuple[RankObjective, ...], requested)


def parse_oof_models(value: str) -> Tuple[OOFModelName, ...]:
    if not value.strip():
        return ()
    requested = tuple(part.strip() for part in value.split(",") if part.strip())
    invalid = sorted(set(requested) - set(VALID_OOF_MODELS))
    if invalid:
        raise ValueError(
            f"Unknown OOF model names: {invalid}. "
            f"Valid choices are: {VALID_OOF_MODELS}"
        )
    return cast(Tuple[OOFModelName, ...], tuple(dict.fromkeys(requested)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train an Expedia learning-to-rank ensemble and create a submission."
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("data/processed/training_set_VU_DM.parquet"),
        help="Path to the processed training parquet file.",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=Path("data/processed/test_set_VU_DM.parquet"),
        help="Path to the processed test parquet file.",
    )
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--validation-predictions",
        type=Path,
        default=Path("validation_predictions.csv"),
        help="Where to write validation predictions. Use 'none' to disable.",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=Path("metrics.json"),
        help="Where to write validation metrics. Use 'none' to disable.",
    )
    parser.add_argument(
        "--split-strategy",
        choices=VALID_SPLIT_STRATEGIES,
        default="time",
        help="Validation split strategy. All choices keep full srch_id groups intact.",
    )
    parser.add_argument(
        "--validation-splits",
        default="",
        help=(
            "Optional comma-separated validation audit splits, e.g. "
            "time,random,srch_id_quantile. The primary --split-strategy still drives "
            "the final model iteration."
        ),
    )
    parser.add_argument("--valid-fraction", type=float, default=0.15)
    parser.add_argument(
        "--workflow",
        choices=VALID_WORKFLOWS,
        default="oof",
        help=(
            "Training workflow. 'oof' runs grouped out-of-fold CV and ensembling; "
            "'legacy' keeps the previous single holdout LightGBM path."
        ),
    )
    parser.add_argument(
        "--oof-folds",
        type=int,
        default=5,
        help=(
            "Number of srch_id-grouped OOF folds. Folds are stratified by "
            "random_bool, month, and whether the query has a booking when possible."
        ),
    )
    parser.add_argument(
        "--oof-models",
        default="lgbm_ranker,lgbm_booking,lgbm_click,xgb_ranker,catboost_booking",
        help=(
            "Comma-separated OOF base models. Optional models are skipped with a "
            "warning if their dependency is unavailable."
        ),
    )
    parser.add_argument(
        "--ensemble-weight-step",
        type=float,
        default=0.05,
        help="Grid step for constrained rank-average OOF ensemble weights.",
    )
    parser.add_argument(
        "--no-meta-ranker",
        action="store_true",
        help="Disable the small OOF-stacked LightGBM meta-ranker.",
    )
    parser.add_argument(
        "--stacking-min-delta",
        type=float,
        default=0.003,
        help="Minimum OOF NDCG@5 gain required before using stacked predictions.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--target-encoding-folds", type=int, default=5)
    parser.add_argument("--target-encoding-smoothing", type=float, default=40.0)
    parser.add_argument("--n-estimators", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=96)
    parser.add_argument("--max-depth", type=int, default=-1)
    parser.add_argument("--min-child-samples", type=int, default=2000)
    parser.add_argument("--subsample", type=float, default=0.85)
    parser.add_argument("--colsample-bytree", type=float, default=0.85)
    parser.add_argument("--reg-alpha", type=float, default=0.1)
    parser.add_argument("--reg-lambda", type=float, default=2.0)
    parser.add_argument("--lambdarank-truncation-level", type=int, default=8)
    parser.add_argument("--early-stopping-rounds", type=int, default=150)
    parser.add_argument(
        "--rank-objective",
        choices=VALID_RANK_OBJECTIVES,
        default="lambdarank",
        help="Primary LightGBM ranker objective.",
    )
    parser.add_argument(
        "--ranker-blend-seeds",
        default="",
        help=(
            "Legacy workflow only. Comma-separated extra LightGBM random seeds for "
            "same-feature ranker blending. Empty disables ranker blending."
        ),
    )
    parser.add_argument(
        "--ranker-blend-objectives",
        default="",
        help=(
            "Legacy workflow only. Comma-separated extra ranker objectives to train "
            "after the shared feature build, e.g. rank_xendcg."
        ),
    )
    parser.add_argument(
        "--ranker-blend-mode",
        choices=VALID_RANKER_BLEND_MODES,
        default="raw",
        help="Blend ranker scores as raw scores or within-search percentile ranks.",
    )
    parser.add_argument(
        "--ranker-blend-weight-step",
        type=float,
        default=0.05,
        help=(
            "Validation grid step for automatic ranker blend weight tuning. "
            "For example, 0.05 tests weights in 5 percentage point increments."
        ),
    )
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--sample-groups",
        type=int,
        default=0,
        help="Sample this many srch_id groups for smoke tests. 0 uses all data.",
    )
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--no-final", action="store_true")
    parser.add_argument(
        "--use-raw-prop-id",
        action="store_true",
        help=(
            "Legacy workflow only. Include raw prop_id as a LightGBM categorical "
            "feature. Off by default."
        ),
    )
    parser.add_argument(
        "--no-aux-position-model",
        action="store_true",
        help=(
            "Disable the auxiliary learned position-model feature. Historical "
            "position aggregate features are still used when available."
        ),
    )
    parser.add_argument(
        "--aux-position-model-path",
        type=Path,
        default=None,
        help=(
            "Path to an existing auxiliary position model pickle. When set, the "
            "model is loaded and used to create position_model_log1p_pred instead "
            "of fitting a new auxiliary position model."
        ),
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Log detailed stage timings.",
    )
    return parser


def none_path(value: Path) -> Optional[Path]:
    if str(value).lower() == "none":
        return None
    return value


def parse_args(argv: Optional[Sequence[str]] = None) -> Config:
    parser = build_parser()
    args = parser.parse_args(argv)

    validation_predictions = none_path(args.validation_predictions)
    metrics_path = none_path(args.metrics_path)
    validation_splits = parse_validation_splits(args.validation_splits)
    ranker_blend_seeds = parse_ranker_blend_seeds(args.ranker_blend_seeds)
    ranker_blend_objectives = parse_rank_objectives(args.ranker_blend_objectives)
    oof_models = parse_oof_models(args.oof_models)

    return Config(
        train_path=args.train,
        test_path=args.test,
        output_path=args.output,
        model_dir=args.model_dir,
        validation_predictions_path=validation_predictions,
        metrics_path=metrics_path,
        split_strategy=args.split_strategy,
        validation_splits=validation_splits,
        valid_fraction=args.valid_fraction,
        workflow=cast(Workflow, args.workflow),
        oof_folds=args.oof_folds,
        oof_models=oof_models,
        ensemble_weight_step=args.ensemble_weight_step,
        use_meta_ranker=not args.no_meta_ranker,
        stacking_min_delta=args.stacking_min_delta,
        seed=args.seed,
        target_encoding_folds=args.target_encoding_folds,
        target_encoding_smoothing=args.target_encoding_smoothing,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        max_depth=args.max_depth,
        min_child_samples=args.min_child_samples,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        lambdarank_truncation_level=args.lambdarank_truncation_level,
        early_stopping_rounds=args.early_stopping_rounds,
        rank_objective=cast(RankObjective, args.rank_objective),
        ranker_blend_seeds=ranker_blend_seeds,
        ranker_blend_objectives=ranker_blend_objectives,
        ranker_blend_weight_step=args.ranker_blend_weight_step,
        ranker_blend_mode=cast(RankerBlendMode, args.ranker_blend_mode),
        n_jobs=args.n_jobs,
        sample_groups=args.sample_groups,
        skip_validation=args.skip_validation,
        no_final=args.no_final,
        use_raw_prop_id=args.use_raw_prop_id,
        use_aux_position_model=not args.no_aux_position_model,
        aux_position_model_path=args.aux_position_model_path,
        profile=args.profile,
    )


def configure_logging() -> None:
    # Ignore performance warnings from pandas
    warnings.filterwarnings("ignore", category=PerformanceWarning)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
