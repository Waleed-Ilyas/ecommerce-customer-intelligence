"""Monthly acquisition-cohort retention (aggregations written in SQL with DuckDB)."""
from __future__ import annotations

import duckdb
import pandas as pd

COHORT_SQL = """
WITH inv AS (
    SELECT customer_id, date_trunc('month', date) AS month, revenue_capped FROM orders
),
first_month AS (
    SELECT customer_id, min(month) AS cohort FROM inv GROUP BY customer_id
)
SELECT f.cohort, date_diff('month', f.cohort, i.month) AS age,
       count(DISTINCT i.customer_id) AS customers, sum(i.revenue_capped) AS revenue
FROM inv i JOIN first_month f USING (customer_id)
GROUP BY 1, 2
ORDER BY 1, 2
"""


def cohort_tables(invoices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Cohort = month of a customer's first purchase. Returns customer counts, retention % and
    revenue by (cohort, months since first purchase)."""
    con = duckdb.connect()
    con.register("orders", invoices[["customer_id", "date", "revenue_capped"]])
    long = con.execute(COHORT_SQL).df()
    long["cohort"] = pd.PeriodIndex(long["cohort"], freq="M")
    active = long.pivot(index="cohort", columns="age", values="customers")
    revenue = long.pivot(index="cohort", columns="age", values="revenue")
    size = active[0]
    retention = active.div(size, axis=0) * 100
    return {"active": active, "retention_pct": retention, "revenue": revenue,
            "cohort_size": size.to_frame("customers")}
