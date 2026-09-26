"""Snapshot features and labels.

For a snapshot date T, features use only invoices with date <= T. Labels use invoices in
(T, T + horizon]. Stacking several snapshots gives a time-aware training set: no row ever
sees information from after its own snapshot.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import btyd
from . import config as C

BTYD_FEATURES = ["btyd_exp_purchases", "btyd_p_alive", "btyd_order_value", "btyd_clv"]
CALENDAR_FEATURES = ["q4_window_share"]  # share of the forecast window that falls in Oct-Dec

FEATURES = [
    "recency_days", "tenure_days", "n_orders", "n_order_days", "monetary_total",
    "avg_order_value", "std_order_value", "avg_gap_days", "overdue_ratio", "orders_30d",
    "orders_90d", "orders_180d", "revenue_30d", "revenue_90d", "revenue_180d",
    "revenue_prev_180d", "revenue_trend", "avg_items_per_order", "avg_units_per_line",
    "n_distinct_products", "return_rate", "is_uk", "share_q4_orders",
]


def q4_window_share(snapshot: pd.Timestamp, horizon_days: int = C.HORIZON_DAYS) -> float:
    """Known-in-advance seasonality: fraction of the coming window that is Oct-Dec (peak season)."""
    days = pd.date_range(snapshot + pd.Timedelta(days=1), periods=horizon_days, freq="D")
    return float(days.month.isin([10, 11, 12]).mean())


def snapshot_features(
    invoices: pd.DataFrame, returns: pd.DataFrame, sales: pd.DataFrame, snapshot: pd.Timestamp
) -> pd.DataFrame:
    """One row per customer that has purchased on or before `snapshot`."""
    snap = pd.Timestamp(snapshot).normalize()
    inv = invoices[invoices["date"] <= snap]
    g = inv.groupby("customer_id")

    f = pd.DataFrame({
        "first_date": g["date"].min(),
        "last_date": g["date"].max(),
        "n_orders": g.size(),
        "n_order_days": g["date"].nunique(),
        "monetary_total": g["revenue_capped"].sum(),
        "avg_order_value": g["revenue_capped"].mean(),
        "std_order_value": g["revenue_capped"].std().fillna(0.0),
        "avg_items_per_order": g["n_lines"].mean(),
        "is_uk": g["country"].agg(lambda s: float((s == "United Kingdom").mean())),
    })
    f["recency_days"] = (snap - f["last_date"]).dt.days
    f["tenure_days"] = (snap - f["first_date"]).dt.days
    # average gap between purchase days; unknown for one-off buyers -> use tenure as a proxy
    f["avg_gap_days"] = np.where(f["n_order_days"] > 1,
                                 (f["last_date"] - f["first_date"]).dt.days /
                                 (f["n_order_days"] - 1).clip(lower=1), np.nan)
    f["overdue_ratio"] = f["recency_days"] / f["avg_gap_days"].replace(0, np.nan)

    for days in (30, 90, 180):
        recent = inv[inv["date"] > snap - pd.Timedelta(days=days)].groupby("customer_id")
        f[f"orders_{days}d"] = recent.size().reindex(f.index).fillna(0)
        f[f"revenue_{days}d"] = recent["revenue_capped"].sum().reindex(f.index).fillna(0)
    prev = inv[(inv["date"] <= snap - pd.Timedelta(days=180)) &
               (inv["date"] > snap - pd.Timedelta(days=360))]
    f["revenue_prev_180d"] = prev.groupby("customer_id")["revenue_capped"].sum().reindex(
        f.index).fillna(0)
    f["revenue_trend"] = (f["revenue_180d"] - f["revenue_prev_180d"]) / (
        f["revenue_180d"] + f["revenue_prev_180d"] + 1.0)  # bounded in (-1, 1)

    s = sales[sales["invoice_date"].dt.normalize() <= snap].groupby("customer_id")
    f["n_distinct_products"] = s["stock_code"].nunique().reindex(f.index)
    f["avg_units_per_line"] = s["quantity"].mean().reindex(f.index)

    r = returns[returns["invoice_date"].dt.normalize() <= snap].groupby("customer_id")
    f["return_rate"] = (r["line_value"].sum().reindex(f.index).fillna(0) /
                        (g["revenue"].sum() + 1.0)).clip(upper=1.0)
    q4 = inv["date"].dt.month.isin([10, 11, 12])
    f["share_q4_orders"] = q4.groupby(inv["customer_id"]).mean().reindex(f.index)

    f["snapshot"] = snap
    return f.drop(columns=["first_date", "last_date"]).reset_index()


def add_labels(
    feats: pd.DataFrame, invoices: pd.DataFrame, snapshot: pd.Timestamp,
    horizon_days: int = C.HORIZON_DAYS,
) -> pd.DataFrame:
    """Future revenue and churn flag in (snapshot, snapshot + horizon]."""
    snap = pd.Timestamp(snapshot).normalize()
    end = snap + pd.Timedelta(days=horizon_days)
    fut = invoices[(invoices["date"] > snap) & (invoices["date"] <= end)]
    g = fut.groupby("customer_id")
    out = feats.copy()
    out["future_revenue"] = out["customer_id"].map(g["revenue_capped"].sum()).fillna(0.0)
    out["future_orders"] = out["customer_id"].map(g.size()).fillna(0).astype(int)
    out["churned"] = (out["future_orders"] == 0).astype(int)
    return out


def add_btyd_features(feats: pd.DataFrame, invoices: pd.DataFrame, snapshot: pd.Timestamp):
    """Stack the probabilistic BG/NBD + Gamma-Gamma model as features. It is *fitted at the
    snapshot on data up to the snapshot only*, so it carries no future information."""
    snap = pd.Timestamp(snapshot).normalize()
    summ = btyd.summary(invoices, snap)
    model = btyd.BTYDModel().fit(summ)
    first = invoices[invoices["date"] <= snap].sort_values("invoice_date").groupby(
        "customer_id")["revenue_capped"].first()
    pred = model.predict(summ, first).rename_axis("customer_id").reset_index()
    return feats.merge(pred, on="customer_id", how="left")


def build_dataset(
    invoices: pd.DataFrame, returns: pd.DataFrame, sales: pd.DataFrame,
    snapshots: list[str], labelled: bool = True, with_btyd: bool = True,
) -> pd.DataFrame:
    parts = []
    for s in snapshots:
        snap = pd.Timestamp(s)
        f = snapshot_features(invoices, returns, sales, snap)
        f["q4_window_share"] = q4_window_share(snap)
        if with_btyd:
            f = add_btyd_features(f, invoices, snap)
        parts.append(add_labels(f, invoices, snap) if labelled else f)
    return pd.concat(parts, ignore_index=True)
