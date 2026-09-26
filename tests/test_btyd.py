import numpy as np
import pandas as pd

from custintel import btyd


def test_posterior_matches_lifetimes_and_is_finite(cleaned):
    _, _, _, inv = cleaned
    s = btyd.summary(inv, pd.Timestamp("2010-12-31"))
    m = btyd.BTYDModel().fit(s)
    e, alive = btyd.posterior_purchases(dict(m.bgf.params_), s.frequency, s.recency, s["T"], 180)
    assert np.isfinite(e).all() and np.isfinite(alive).all()
    assert ((alive >= 0) & (alive <= 1)).all() and (e >= 0).all()
    ref = np.asarray(m.bgf.conditional_probability_alive(s.frequency, s.recency, s["T"]))
    assert np.allclose(alive, ref, atol=1e-6)


def test_predict_columns(cleaned):
    _, _, _, inv = cleaned
    s = btyd.summary(inv, pd.Timestamp("2010-12-31"))
    first = inv.sort_values("invoice_date").groupby("customer_id")["revenue_capped"].first()
    out = btyd.BTYDModel().fit(s).predict(s, first)
    assert {"btyd_exp_purchases", "btyd_p_alive", "btyd_order_value", "btyd_clv"} <= set(out)
    assert (out["btyd_clv"] >= 0).all()
