"""Report figures (matplotlib, dark/gold theme matching the portfolio site)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

BG, PANEL, IVORY, GOLD, BRONZE, CYAN, RED = (
    "#0A0A0C", "#15110F", "#F2EDE4", "#D9B26A", "#8B5E3C", "#4FD1C5", "#ff5a4f",
)
SEGMENT_COLORS = {
    "Champions": GOLD, "Loyal": CYAN, "New Customers": "#9ad1a0", "At Risk": "#ff9f5a",
    "Lost": "#7d7368", "Hibernating": "#a58b6f", "Potential Loyalists": "#c9b8e8",
}


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
        "axes.edgecolor": "#3a332d", "axes.labelcolor": IVORY, "text.color": IVORY,
        "xtick.color": IVORY, "ytick.color": IVORY, "grid.color": "#2a2521", "grid.linewidth": 0.8,
        "axes.grid": True, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold", "legend.frameon": False,
        "figure.dpi": 110, "font.family": "DejaVu Sans",
    })


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cleaning_waterfall(report: dict, path) -> None:
    apply_style()
    steps = report["steps"]
    labels = ["Raw rows"] + [s["step"].split(" (")[0] for s in steps]
    removed = [report["raw_rows"]] + [s["rows_removed"] for s in steps]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    colors = [CYAN] + [BRONZE] * len(steps)
    bars = ax.barh(labels[::-1], removed[::-1], color=colors[::-1])
    ax.bar_label(bars, labels=[f"{v:,}" for v in removed[::-1]], color=IVORY, padding=4, fontsize=9)
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlabel("rows (log scale)")
    ax.set_title(f"Cleaning waterfall: {report['raw_rows']:,} raw lines -> "
                 f"{report['clean_sales_lines']:,} usable customer purchase lines")
    ax.set_xlim(0, report["raw_rows"] * 4)
    _save(fig, path)


def plot_cohorts(retention: pd.DataFrame, sizes: pd.Series, path) -> None:
    apply_style()
    r = retention.copy()
    r = r[[c for c in r.columns if int(c) <= 12]]
    fig, ax = plt.subplots(figsize=(12, 7.5))
    im = ax.imshow(r.values, aspect="auto", cmap="YlOrBr", vmin=0, vmax=60)
    ax.set_yticks(range(len(r)), [f"{i}  (n={int(sizes.loc[i])})" for i in r.index], fontsize=8)
    ax.set_xticks(range(r.shape[1]), r.columns)
    ax.set_xlabel("months since first purchase")
    ax.set_title("Monthly acquisition cohorts: % of customers who order again (month 0 = 100%)")
    ax.grid(False)
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            v = r.values[i, j]
            if not np.isnan(v) and j > 0:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7,
                        color=IVORY if v > 33 else BG)
    fig.colorbar(im, ax=ax, label="% retained")
    _save(fig, path)


def plot_k_selection(k_table: pd.DataFrame, chosen: int, path) -> None:
    apply_style()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(k_table.k, k_table.inertia, marker="o", color=GOLD, lw=2.5)
    ax[0].axvline(chosen, color=CYAN, ls="--")
    ax[0].set_title("Elbow (inertia)")
    ax[0].set_xlabel("k")
    ax[1].plot(k_table.k, k_table.silhouette, marker="o", color=GOLD, lw=2.5)
    ax[1].axvline(chosen, color=CYAN, ls="--")
    ax[1].set_title("Silhouette (sampled)")
    ax[1].set_xlabel("k")
    _save(fig, path)


def plot_segments(customers: pd.DataFrame, summary: pd.DataFrame, path) -> None:
    apply_style()
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1.5, 1]})
    for seg, g in customers.groupby("segment"):
        ax[0].scatter(g["recency_days"], g["monetary_total"], s=9, alpha=0.5, edgecolor="none",
                      color=SEGMENT_COLORS.get(seg.split(" (")[0], IVORY), label=seg)
    ax[0].set_yscale("log")
    ax[0].set_xlabel("days since last order (recency)")
    ax[0].set_ylabel("lifetime spend (GBP, log)")
    ax[0].set_title("RFM K-Means segments")
    ax[0].legend(markerscale=2.5, loc="upper right")
    s = summary.sort_values("revenue_share_pct")
    y = np.arange(len(s))
    ax[1].barh(y - 0.2, s["customer_share_pct"], 0.4, color=CYAN, label="% of customers")
    ax[1].barh(y + 0.2, s["revenue_share_pct"], 0.4, color=GOLD, label="% of revenue")
    ax[1].set_yticks(y, s["segment"])
    ax[1].set_title("Who drives revenue?")
    ax[1].legend(loc="lower right")
    _save(fig, path)


def plot_churn_roc(pred: pd.DataFrame, path, names: dict[str, tuple[str, str]]) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for col, (label, color) in names.items():
        fpr, tpr, _ = roc_curve(pred["churned"], pred[col])
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(pred["churned"], pred[col])
        ax.plot(fpr, tpr, color=color, lw=2.2, label=f"{label} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#5a5148", ls=":")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("Churn: test snapshot 2011-06-09")
    ax.legend(loc="lower right")
    _save(fig, path)


def plot_clv_gains(pred: pd.DataFrame, path, cols: dict[str, tuple[str, str]]) -> None:
    """Cumulative-gains chart: share of the next 6 months' revenue found in the top x% of customers."""
    apply_style()
    fig, ax = plt.subplots(figsize=(7.5, 5.4))
    y = pred["future_revenue"].to_numpy()
    x = np.arange(1, len(y) + 1) / len(y) * 100
    ideal = np.cumsum(np.sort(y)[::-1]) / y.sum() * 100
    ax.plot(x, ideal, color=IVORY, ls=":", lw=1.5, label="Perfect ranking")
    for col, (label, color) in cols.items():
        order = np.argsort(-pred[col].to_numpy())
        ax.plot(x, np.cumsum(y[order]) / y.sum() * 100, color=color, lw=2.4, label=label)
    ax.plot([0, 100], [0, 100], color="#5a5148", ls="--", lw=1)
    ax.set_xlabel("top % of customers (ranked by predicted CLV)")
    ax.set_ylabel("% of actual 6-month revenue captured")
    ax.set_title("CLV ranking quality on the test snapshot")
    ax.legend(loc="lower right")
    _save(fig, path)


def plot_shap_bars(shap_top: dict, title: str, path) -> None:
    apply_style()
    s = pd.Series(shap_top).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(s.index, s.values, color=GOLD)
    ax.set_xlabel("mean |SHAP|")
    ax.set_title(title)
    _save(fig, path)
