import numpy as np
import pandas as pd

from custintel import cohort, explain, segment
from custintel import features as F
from custintel import models as M
from custintel import train as T


def test_persona_rules():
    prof = pd.DataFrame({"recency_days": [7, 30, 26, 254, 413],
                         "n_orders": [24, 9, 2, 4, 1],
                         "tenure_days": [725, 631, 143, 592, 444]})
    names = segment.name_personas(prof)
    assert list(names.values()) == ["Champions", "Loyal", "New Customers", "At Risk", "Lost"]


def test_rfm_scores_range(cleaned):
    sales, returns, _, inv = cleaned
    f = F.snapshot_features(inv, returns, sales, pd.Timestamp("2010-12-31"))
    r = segment.rfm_scores(f)
    assert r[["R", "F", "M"]].min().min() == 1 and r[["R", "F", "M"]].max().max() == 5


def test_cohort_month_zero_is_full(cleaned):
    _, _, _, inv = cleaned
    t = cohort.cohort_tables(inv)
    assert np.allclose(t["retention_pct"][0], 100)
    assert t["cohort_size"]["customers"].sum() == inv["customer_id"].nunique()


def test_metrics_and_selection_helpers():
    y = np.array([0, 0, 1, 1, 1])
    assert M.churn_metrics(y, np.array([0.1, 0.2, 0.7, 0.8, 0.9]))["roc_auc"] == 1.0
    rev = np.array([0, 0, 10, 50, 200.0])
    m = M.clv_metrics(rev, rev)
    assert m["mae"] == 0 and m["spearman"] > 0.99 and m["top_decile_revenue_capture_pct"] > 75
    scores = {"a": 0.80, "b": 0.805, "c": 0.90}
    assert T.select_simplest(["a", "b", "c"], scores, margin=0.01) == "c"
    assert T.select_simplest(["a", "b", "c"], {"a": 0.89, "b": 0.895, "c": 0.90}, 0.02) == "a"


def test_paired_bootstrap_identical_scores_has_zero_diff():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    s = rng.random(300)
    r = T.paired_bootstrap(y, s, s, T._auc, n=50)
    assert r["diff"] == 0 and r["ci95"] == [0.0, 0.0]


def test_signals_text():
    f = pd.DataFrame({"recency_days": [300, 5], "orders_90d": [0, 4], "n_order_days": [3, 12],
                      "overdue_ratio": [4.0, 0.2], "revenue_trend": [-0.5, 0.6],
                      "return_rate": [0.0, 0.0], "n_orders": [3, 12]})
    s = explain.signals(f)
    assert "no order for 300 days" in s[0] and "habitual buyer" in s[1]
