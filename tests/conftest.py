import numpy as np
import pandas as pd
import pytest

from custintel import clean


@pytest.fixture(scope="session")
def raw() -> pd.DataFrame:
    """Small synthetic raw transaction table containing every kind of mess the real one has."""
    rng = np.random.default_rng(0)
    rows, inv_no = [], 100000
    for cust in range(1, 121):
        n_orders = int(rng.integers(1, 9))
        start = pd.Timestamp("2010-01-04") + pd.Timedelta(days=int(rng.integers(0, 300)))
        for k in range(n_orders):
            date = start + pd.Timedelta(days=int(k * rng.integers(15, 60)), hours=10)
            inv_no += 1
            for _ in range(int(rng.integers(1, 5))):
                rows.append((str(inv_no), f"P{rng.integers(1, 40)}", "WIDGET", int(rng.integers(1, 12)),
                             date, round(float(rng.uniform(1, 9)), 2), cust, "United Kingdom"))
    cols = ["invoice", "stock_code", "description", "quantity", "invoice_date", "price",
            "customer_id", "country"]
    df = pd.DataFrame(rows, columns=cols)
    day = pd.Timestamp("2010-03-01")
    extras = [
        ("A900001", "B", "Adjust bad debt", 1, day, -50.0, pd.NA, "United Kingdom"),
        ("200001", "POST", "POSTAGE", 1, day, 15.0, 5, "United Kingdom"),
        ("200002", "P1", "FREE", 1, day, 0.0, 5, "United Kingdom"),
        ("C300001", "P2", "WIDGET", -2, pd.Timestamp("2010-04-01"), 5.0, 5, "United Kingdom"),
        ("200003", "P3", "GUEST", 3, day, 2.0, pd.NA, "France"),
    ]
    messy = pd.concat([df, df.iloc[:5], pd.DataFrame(extras, columns=cols)], ignore_index=True)
    messy["customer_id"] = messy["customer_id"].astype("Int64")
    for c in ("invoice", "stock_code", "description", "country"):
        messy[c] = messy[c].astype("string")
    return messy


@pytest.fixture(scope="session")
def cleaned(raw):
    sales, returns, report = clean.clean(raw)
    return sales, returns, report, clean.to_invoices(sales)
