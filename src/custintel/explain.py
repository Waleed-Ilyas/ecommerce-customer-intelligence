"""Plain-language risk signals for the customer lookup.

These are transparent business rules on the customer's own history (not model attributions):
logistic-regression coefficients on correlated RFM features can carry counter-intuitive signs for an
individual, so the app shows facts a marketer can verify at a glance next to the model's score.
"""
from __future__ import annotations

import pandas as pd


def signals(f: pd.DataFrame) -> pd.Series:
    """One string per customer, e.g. 'RISK: no order for 254d; 3.6x longer than usual | OK: ...'."""
    out = []
    for r in f.itertuples(index=False):
        risk, good = [], []
        if r.recency_days > 180:
            risk.append(f"no order for {int(r.recency_days)} days")
        elif r.orders_90d == 0:
            risk.append("no orders in the last 90 days")
        if r.n_order_days > 1 and pd.notna(r.overdue_ratio) and r.overdue_ratio > 2:
            risk.append(f"{r.overdue_ratio:.1f}x longer than their usual gap since last order")
        if r.revenue_trend < -0.3:
            risk.append("spend down vs the previous 6 months")
        if r.return_rate > 0.1:
            risk.append("high return rate")
        if r.n_orders == 1:
            risk.append("only one order so far")
        if r.recency_days <= 30:
            good.append("ordered in the last 30 days")
        if r.orders_90d >= 3:
            good.append("3+ orders in the last 90 days")
        if r.revenue_trend > 0.3:
            good.append("spend up vs the previous 6 months")
        if r.n_order_days >= 10:
            good.append("habitual buyer (10+ order days)")
        out.append(" | ".join(filter(None, [
            "Risk signals: " + "; ".join(risk) if risk else "",
            "Healthy signals: " + "; ".join(good) if good else ""])) or "No strong signals")
    return pd.Series(out, index=f.index)
