from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PLOTS_DIR = ROOT / "plots"
ARTIFACTS_DIR = ROOT / "artifacts"

GROUP_COL = "srch_id"
ITEM_COL = "prop_id"
TARGET_COL = "relevance"
STEP = 0.05
K = 5

LGBM_PREDICTIONS = ROOT / "validation_predictions_oof_lgbm.csv"
XGB_PREDICTIONS = ROOT / "validation_predictions_oof_xgb.csv"
PLOT_PATH = PLOTS_DIR / "blend_weight_performance.png"
LGBM_GRID_PATH = ARTIFACTS_DIR / "lgbm_blend_weight_grid.csv"
FINAL_GRID_PATH = ARTIFACTS_DIR / "final_lgbm_xgb_weight_grid.csv"

LGBM_SCORE_COLS = [
    "lgbm_ranker_score",
    "lgbm_click_score",
    "lgbm_booking_score",
]


def integer_weight_compositions(total: int, parts: int):
    if parts == 1:
        yield (total,)
        return
    for value in range(total + 1):
        for rest in integer_weight_compositions(total - value, parts - 1):
            yield (value, *rest)


def blend_weight_grid(parts: int, step: float = STEP) -> list[tuple[float, ...]]:
    units = int(round(1.0 / step))
    return [
        tuple(value / units for value in values)
        for values in integer_weight_compositions(units, parts)
    ]


def add_group_rank_pct(df: pd.DataFrame, cols: list[str]) -> list[str]:
    rank_cols = []
    for col in cols:
        rank_col = f"{col}_rank_pct"
        df[rank_col] = (
            df.groupby(GROUP_COL, sort=False)[col]
            .rank(method="average", pct=True)
            .astype("float32")
        )
        rank_cols.append(rank_col)
    return rank_cols


def load_predictions() -> pd.DataFrame:
    lgbm = pd.read_csv(
        LGBM_PREDICTIONS,
        usecols=[
            GROUP_COL,
            ITEM_COL,
            TARGET_COL,
            "lgbm_ranker_score",
            "lgbm_booking_score",
            "lgbm_click_score",
            "ensemble_rankavg_score",
        ],
        dtype={
            GROUP_COL: "int32",
            ITEM_COL: "int32",
            TARGET_COL: "int8",
            "lgbm_ranker_score": "float32",
            "lgbm_booking_score": "float32",
            "lgbm_click_score": "float32",
            "ensemble_rankavg_score": "float32",
        },
    )
    xgb = pd.read_csv(
        XGB_PREDICTIONS,
        usecols=[GROUP_COL, ITEM_COL, "xgb_ranker_score"],
        dtype={
            GROUP_COL: "int32",
            ITEM_COL: "int32",
            "xgb_ranker_score": "float32",
        },
    )

    same_order = (
        len(lgbm) == len(xgb)
        and np.array_equal(lgbm[GROUP_COL].to_numpy(), xgb[GROUP_COL].to_numpy())
        and np.array_equal(lgbm[ITEM_COL].to_numpy(), xgb[ITEM_COL].to_numpy())
    )
    if same_order:
        lgbm["xgb_ranker_score"] = xgb["xgb_ranker_score"].to_numpy(dtype="float32")
        return lgbm

    return lgbm.merge(xgb, on=[GROUP_COL, ITEM_COL], how="inner", validate="one_to_one")


def make_padded_arrays(
    df: pd.DataFrame,
    score_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], np.ndarray]:
    order = np.argsort(df[GROUP_COL].to_numpy(), kind="mergesort")
    groups = df[GROUP_COL].to_numpy()[order]
    relevance = df[TARGET_COL].to_numpy(dtype="int8")[order]

    _, starts, counts = np.unique(groups, return_index=True, return_counts=True)
    group_count = len(counts)
    max_group_size = int(counts.max())

    row_idx = np.repeat(np.arange(group_count, dtype="int32"), counts)
    col_idx = np.arange(len(groups), dtype="int32") - np.repeat(starts, counts)

    relevance_pad = np.zeros((group_count, max_group_size), dtype="int8")
    relevance_pad[row_idx, col_idx] = relevance
    valid_mask = np.zeros((group_count, max_group_size), dtype=bool)
    valid_mask[row_idx, col_idx] = True

    score_pads: dict[str, np.ndarray] = {}
    for col in score_cols:
        values = df[col].to_numpy(dtype="float32")[order]
        pad = np.zeros((group_count, max_group_size), dtype="float32")
        pad[row_idx, col_idx] = values
        score_pads[col] = pad

    idcg = idcg_at_k(relevance_pad, K)
    return relevance_pad, idcg, score_pads, valid_mask


def idcg_at_k(relevance_pad: np.ndarray, k: int) -> np.ndarray:
    k = min(k, relevance_pad.shape[1])
    ideal_idx = np.argsort(-relevance_pad, axis=1, kind="stable")[:, :k]
    ideal_rel = np.take_along_axis(relevance_pad, ideal_idx, axis=1).astype("float32")
    discounts = (1.0 / np.log2(np.arange(2, k + 2, dtype="float32"))).reshape(1, -1)
    return ((np.exp2(ideal_rel) - 1.0) * discounts).sum(axis=1)


def ndcg_at_k_from_scores(
    relevance_pad: np.ndarray,
    idcg: np.ndarray,
    scores: np.ndarray,
    k: int,
) -> float:
    k = min(k, scores.shape[1])
    ranked_idx = np.argsort(-scores, axis=1, kind="stable")[:, :k]
    rel = np.take_along_axis(relevance_pad, ranked_idx, axis=1).astype("float32")
    discounts = (1.0 / np.log2(np.arange(2, k + 2, dtype="float32"))).reshape(1, -1)
    dcg = ((np.exp2(rel) - 1.0) * discounts).sum(axis=1)
    ndcg = np.zeros_like(dcg, dtype="float32")
    np.divide(dcg, idcg, out=ndcg, where=idcg > 0)
    return float(ndcg.mean(dtype="float64"))


def compute_lgbm_grid(
    relevance_pad: np.ndarray,
    idcg: np.ndarray,
    score_pads: dict[str, np.ndarray],
    valid_mask: np.ndarray,
) -> pd.DataFrame:
    records = []
    for ranker_w, click_w, booking_w in blend_weight_grid(3):
        scores = (
            np.float32(ranker_w) * score_pads["lgbm_ranker_score_rank_pct"]
            + np.float32(click_w) * score_pads["lgbm_click_score_rank_pct"]
            + np.float32(booking_w) * score_pads["lgbm_booking_score_rank_pct"]
        )
        scores = np.where(valid_mask, scores, -np.inf).astype("float32")
        records.append(
            {
                "ranker_weight": ranker_w,
                "click_weight": click_w,
                "booking_weight": booking_w,
                "oof_ndcg_at_5": ndcg_at_k_from_scores(relevance_pad, idcg, scores, K),
            }
        )
    return pd.DataFrame.from_records(records)


def compute_final_grid(
    relevance_pad: np.ndarray,
    idcg: np.ndarray,
    score_pads: dict[str, np.ndarray],
    valid_mask: np.ndarray,
) -> pd.DataFrame:
    records = []
    for xgb_w in np.arange(0.0, 1.0 + STEP / 2.0, STEP):
        xgb_w = round(float(xgb_w), 2)
        lgbm_w = round(1.0 - xgb_w, 2)
        scores = (
            np.float32(lgbm_w) * score_pads["ensemble_rankavg_score_rank_pct"]
            + np.float32(xgb_w) * score_pads["xgb_ranker_score_rank_pct"]
        )
        scores = np.where(valid_mask, scores, -np.inf).astype("float32")
        records.append(
            {
                "lgbm_weight": lgbm_w,
                "xgb_weight": xgb_w,
                "oof_ndcg_at_5": ndcg_at_k_from_scores(relevance_pad, idcg, scores, K),
            }
        )
    return pd.DataFrame.from_records(records)


def plot_results(lgbm_grid: pd.DataFrame, final_grid: pd.DataFrame) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    selected_lgbm = (0.55, 0.30, 0.15)
    selected_final_xgb = 0.25
    lgbm_best = lgbm_grid.loc[lgbm_grid["oof_ndcg_at_5"].idxmax()]
    final_best = final_grid.loc[final_grid["oof_ndcg_at_5"].idxmax()]

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6), constrained_layout=True)

    scatter = axes[0].scatter(
        lgbm_grid["ranker_weight"],
        lgbm_grid["click_weight"],
        c=lgbm_grid["oof_ndcg_at_5"],
        cmap="viridis",
        s=72,
        marker="s",
        edgecolors="none",
    )
    axes[0].scatter(
        [selected_lgbm[0]],
        [selected_lgbm[1]],
        marker="*",
        s=240,
        color="#c62828",
        edgecolor="white",
        linewidth=0.9,
        zorder=3,
        label="Selected 0.55 / 0.30 / 0.15",
    )
    axes[0].set_title("LightGBM internal blend")
    axes[0].set_xlabel("Ranker weight")
    axes[0].set_ylabel("Click weight")
    axes[0].set_xlim(-0.03, 1.03)
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].grid(alpha=0.22)
    axes[0].legend(loc="upper right", frameon=False, fontsize=8)
    axes[0].text(
        0.02,
        0.98,
        f"Best: {lgbm_best['ranker_weight']:.2f} / "
        f"{lgbm_best['click_weight']:.2f} / {lgbm_best['booking_weight']:.2f}\n"
        f"NDCG@5={lgbm_best['oof_ndcg_at_5']:.6f}",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
    )
    cbar = fig.colorbar(scatter, ax=axes[0])
    cbar.set_label("OOF NDCG@5")

    axes[1].plot(
        final_grid["xgb_weight"],
        final_grid["oof_ndcg_at_5"],
        color="#1565c0",
        linewidth=2.2,
        marker="o",
        markersize=4,
    )
    selected_row = final_grid.loc[
        np.isclose(final_grid["xgb_weight"], selected_final_xgb)
    ].iloc[0]
    axes[1].scatter(
        [selected_row["xgb_weight"]],
        [selected_row["oof_ndcg_at_5"]],
        marker="*",
        s=260,
        color="#c62828",
        edgecolor="white",
        linewidth=0.9,
        zorder=3,
        label="Selected 0.75 / 0.25",
    )
    axes[1].set_title("Final LGBM / XGBoost blend")
    axes[1].set_xlabel("XGBoost weight")
    axes[1].set_ylabel("OOF NDCG@5")
    axes[1].set_xlim(-0.02, 1.02)
    y_min = final_grid["oof_ndcg_at_5"].min()
    y_max = final_grid["oof_ndcg_at_5"].max()
    margin = max((y_max - y_min) * 0.12, 0.00005)
    axes[1].set_ylim(y_min - margin, y_max + margin)
    axes[1].grid(alpha=0.22)
    axes[1].legend(loc="upper right", frameon=False, fontsize=8)
    axes[1].text(
        0.02,
        0.02,
        f"Best: LGBM {final_best['lgbm_weight']:.2f}, "
        f"XGB {final_best['xgb_weight']:.2f}\n"
        f"NDCG@5={final_best['oof_ndcg_at_5']:.6f}",
        transform=axes[1].transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
    )

    fig.suptitle("Blend weight validation performance", fontsize=13)
    fig.savefig(PLOT_PATH, dpi=220)
    plt.close(fig)


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_predictions()
    rank_cols = add_group_rank_pct(
        df,
        [
            *LGBM_SCORE_COLS,
            "ensemble_rankavg_score",
            "xgb_ranker_score",
        ],
    )
    relevance_pad, idcg, score_pads, valid_mask = make_padded_arrays(df, rank_cols)

    lgbm_grid = compute_lgbm_grid(relevance_pad, idcg, score_pads, valid_mask)
    final_grid = compute_final_grid(relevance_pad, idcg, score_pads, valid_mask)

    lgbm_grid.to_csv(LGBM_GRID_PATH, index=False)
    final_grid.to_csv(FINAL_GRID_PATH, index=False)
    plot_results(lgbm_grid, final_grid)

    lgbm_best = lgbm_grid.loc[lgbm_grid["oof_ndcg_at_5"].idxmax()]
    final_best = final_grid.loc[final_grid["oof_ndcg_at_5"].idxmax()]
    print(f"Wrote {PLOT_PATH}")
    print(
        "Best LightGBM weights: "
        f"ranker={lgbm_best['ranker_weight']:.2f}, "
        f"click={lgbm_best['click_weight']:.2f}, "
        f"booking={lgbm_best['booking_weight']:.2f}, "
        f"NDCG@5={lgbm_best['oof_ndcg_at_5']:.9f}"
    )
    print(
        "Best final weights: "
        f"LGBM={final_best['lgbm_weight']:.2f}, "
        f"XGB={final_best['xgb_weight']:.2f}, "
        f"NDCG@5={final_best['oof_ndcg_at_5']:.9f}"
    )


if __name__ == "__main__":
    main()
