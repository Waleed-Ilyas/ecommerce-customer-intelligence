import numpy as np
import pandas as pd

from custintel import features as F


def _feats(cleaned, snap):
    sales, returns, _, inv = cleaned
    return F.snapshot_features(inv, returns, sales, pd.Timestamp(snap))


def test_features_ignore_the_future(cleaned):
    """Adding a huge order AFTER the snapshot must not change any feature."""
    sales, returns, _, inv = cleaned
    snap = pd.Timestamp("2010-08-01")
    base = F.snapshot_features(inv, returns, sales, snap)
    future = inv.iloc[[0]].copy()
    future["date"] = snap + pd.Timedelta(days=5)
    future["invoice_date"] = future["date"]
    future["revenue"] = future["revenue_capped"] = 1e6
    changed = F.snapshot_features(pd.concat([inv, future]), returns, sales, snap)
    pd.testing.assert_frame_equal(base.reset_index(drop=True), changed.reset_index(drop=True))


def test_only_customers_seen_before_snapshot(cleaned):
    _, _, _, inv = cleaned
    snap = pd.Timestamp("2010-06-01")
    f = _feats(cleaned, snap)
    seen = inv[inv["date"] <= snap]["customer_id"].nunique()
    assert len(f) == seen and (f["recency_days"] >= 0).all()


def test_labels_use_the_following_window_only(cleaned):
    _, _, _, inv = cleaned
    snap = pd.Timestamp("2010-06-01")
    f = F.add_labels(_feats(cleaned, snap), inv, snap, horizon_days=90)
    win = inv[(inv["date"] > snap) & (inv["date"] <= snap + pd.Timedelta(days=90))]
    expect = win.groupby("customer_id")["revenue_capped"].sum()
    got = f.set_index("customer_id")["future_revenue"]
    for cid, val in expect.items():
        if cid in got.index:
            assert np.isclose(got[cid], val)
    assert ((f["future_orders"] == 0) == (f["churned"] == 1)).all()


def test_q4_window_share():
    assert F.q4_window_share(pd.Timestamp("2010-06-09"), 180) > 0.3
    assert F.q4_window_share(pd.Timestamp("2010-12-09"), 180) < 0.2
