"""Cleaning of the raw transaction lines, with an auditable step-by-step waterfall.

Rules (each one is counted in the report so the README can show the impact):
  1. exact duplicate lines are dropped
  2. 'A...' invoices (bad-debt adjustments) are dropped
  3. non-product stock codes (postage, fees, manual entries, samples, ...) are dropped
  4. lines with price <= 0 are dropped (free items / stock adjustments)
  5. cancellations ('C...' invoices) are separated out: they feed the return-rate features
  6. lines without a Customer ID are separated out: kept for revenue totals, unusable for
     customer-level modelling
  7. quantity <= 0 on non-cancellation lines (damage / lost stock write-offs) is dropped
"""
from __future__ import annotations

import pandas as pd

from . import config as C


def _revenue(df: pd.DataFrame) -> float:
    return float((df["quantity"] * df["price"]).sum())


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (sales, returns, report). `sales` = valid product purchases with a customer id."""
    report: dict = {"raw_rows": len(raw), "steps": []}
    df = raw.copy()
    df["stock_code"] = df["stock_code"].str.strip().str.upper()

    def step(name: str, before: pd.DataFrame, after: pd.DataFrame) -> None:
        report["steps"].append({"step": name, "rows_removed": len(before) - len(after),
                                "rows_left": len(after)})

    n = df
    df = df.drop_duplicates()
    step("exact duplicate lines", n, df)

    n = df
    df = df[~df["invoice"].str.startswith("A")]
    step("bad-debt adjustment invoices (A...)", n, df)

    n = df
    df = df[~df["stock_code"].isin(C.NON_PRODUCT_CODES)]
    step("non-product stock codes (postage, fees, manual, samples)", n, df)

    n = df
    df = df[df["price"] > 0]
    step("price <= 0", n, df)

    n = df
    is_cancel = df["invoice"].str.startswith("C")
    cancellations, df = df[is_cancel], df[~is_cancel]
    report["cancellation_lines"] = len(cancellations)
    step("cancellation invoices (C...) moved to the returns table", n, df)

    n = df
    df = df[df["quantity"] > 0]
    step("quantity <= 0 on non-cancellation lines (write-offs)", n, df)

    n = df
    guest = df[df["customer_id"].isna()]
    df = df[df["customer_id"].notna()].copy()
    step("lines without a Customer ID (kept for revenue totals only)", n, df)
    report["guest_lines"] = len(guest)
    report["guest_revenue"] = _revenue(guest)
    report["guest_revenue_share_pct"] = 100 * report["guest_revenue"] / (
        report["guest_revenue"] + _revenue(df))

    cancellations = cancellations[cancellations["customer_id"].notna()].copy()
    for frame in (df, cancellations):
        frame["customer_id"] = frame["customer_id"].astype("int64")
        frame["line_value"] = frame["quantity"] * frame["price"]
        frame["invoice_date"] = pd.to_datetime(frame["invoice_date"])
    cancellations["line_value"] = cancellations["line_value"].abs()

    report.update({
        "clean_sales_lines": len(df),
        "customers": int(df["customer_id"].nunique()),
        "invoices": int(df["invoice"].nunique()),
        "revenue": _revenue(df),
        "start": str(df["invoice_date"].min()),
        "end": str(df["invoice_date"].max()),
        "return_lines_with_customer": len(cancellations),
    })
    return df.reset_index(drop=True), cancellations.reset_index(drop=True), report


def to_invoices(sales: pd.DataFrame) -> pd.DataFrame:
    """One row per invoice (an order), the unit used for all customer-level features."""
    g = sales.groupby("invoice", sort=False)
    inv = g.agg(customer_id=("customer_id", "first"), invoice_date=("invoice_date", "min"),
                revenue=("line_value", "sum"), n_lines=("stock_code", "size"),
                units=("quantity", "sum"), country=("country", "first")).reset_index()
    inv["date"] = inv["invoice_date"].dt.normalize()
    # A handful of single-invoice wholesale/test orders are orders of magnitude larger than the
    # rest; cap invoice revenue at the 99.9th percentile so they do not dominate CLV targets.
    cap = inv["revenue"].quantile(0.999)
    inv["revenue_capped"] = inv["revenue"].clip(upper=cap)
    inv.attrs["revenue_cap"] = float(cap)
    return inv.sort_values(["customer_id", "invoice_date"]).reset_index(drop=True)
