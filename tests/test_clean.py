def test_waterfall_counts(raw, cleaned):
    _, returns, report, _ = cleaned
    steps = {s["step"].split(" (")[0]: s["rows_removed"] for s in report["steps"]}
    assert steps["exact duplicate lines"] == 5
    assert steps["bad-debt adjustment invoices"] == 1
    assert steps["non-product stock codes"] == 1
    assert steps["price <= 0"] == 1
    assert report["cancellation_lines"] == 1 and len(returns) == 1
    assert report["guest_lines"] == 1


def test_clean_sales_are_valid(cleaned):
    sales, _, _, _ = cleaned
    assert (sales["quantity"] > 0).all() and (sales["price"] > 0).all()
    assert sales["customer_id"].notna().all()
    assert not sales["invoice"].str.startswith(("C", "A")).any()


def test_returns_value_is_positive(cleaned):
    _, returns, _, _ = cleaned
    assert (returns["line_value"] > 0).all()


def test_invoice_table_one_row_per_invoice(cleaned):
    sales, _, _, inv = cleaned
    assert inv["invoice"].is_unique and len(inv) == sales["invoice"].nunique()
    assert (inv["revenue_capped"] <= inv["revenue"] + 1e-9).all()
    assert abs(inv["revenue"].sum() - sales["line_value"].sum()) < 1e-6
