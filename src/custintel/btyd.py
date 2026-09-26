"""Probabilistic customer-lifetime-value: BG/NBD (purchase count / alive) + Gamma-Gamma (spend)."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import ConvergenceError, summary_data_from_transaction_data
from scipy.special import gammaln
from scipy.stats import beta as beta_dist

from . import config as C


def summary(invoices: pd.DataFrame, snapshot: pd.Timestamp) -> pd.DataFrame:
    """frequency = repeat purchase *days*, recency/T in days, monetary = mean order value of the
    repeat purchases (lifetimes convention). Only data on/before the snapshot is used."""
    snap = pd.Timestamp(snapshot).normalize()
    tx = invoices.loc[invoices["date"] <= snap, ["customer_id", "date", "revenue_capped"]]
    s = summary_data_from_transaction_data(
        tx, "customer_id", "date", monetary_value_col="revenue_capped",
        observation_period_end=snap, freq="D")
    return s


def posterior_purchases(params: dict, x, tx, T, horizon: float, n_nodes: int = 4000,
                        chunk: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """Posterior expected purchases in the next `horizon` days and P(alive) for every customer.

    Why not lifetimes' own `conditional_expected_number_of_purchases_up_to_time`? With the
    fitted a < 1 (common in retail data) its closed form divides by (a - 1) and returns NaN for
    ~14% of customers. Here the purchase rate lambda is integrated out analytically (Gamma
    conjugacy) and the dropout probability p is integrated numerically on quantile nodes of its
    Beta(a, b) prior. The P(alive) result is validated against lifetimes' closed form in the tests.

    BG/NBD: lambda ~ Gamma(r, rate alpha), p ~ Beta(a, b). For a customer alive at T,
    E[purchases in t | lambda, p] = (1 - exp(-lambda * p * t)) / p.
    """
    r, alpha, a, b = (float(params[k]) for k in ("r", "alpha", "a", "b"))
    nodes = beta_dist.ppf((np.arange(n_nodes) + 0.5) / n_nodes, a, b)
    nodes = np.clip(nodes, 1e-12, 1 - 1e-12)
    log_1p = np.log1p(-nodes)
    x, tx, T = (np.asarray(v, float) for v in (x, tx, T))
    out_e, out_alive = np.empty(len(x)), np.empty(len(x))
    for i in range(0, len(x), chunk):
        xs, txs, Ts = x[i:i + chunk, None], tx[i:i + chunk, None], T[i:i + chunk, None]
        k = r + xs
        const = r * np.log(alpha) + gammaln(k) - gammaln(r)
        log_a = const - k * np.log(alpha + Ts)  # lambda integrated out, alive term (per p node)
        w_alive = np.exp(xs * log_1p + log_a)
        log_b = const - k * np.log(alpha + txs)
        w_dead = np.where(xs > 0, nodes * np.exp(np.maximum(xs - 1, 0) * log_1p + log_b), 0.0)
        gain = -np.expm1(-k * np.log1p(nodes * horizon / (alpha + Ts))) / nodes
        alive_mass, dead_mass = w_alive.mean(axis=1), w_dead.mean(axis=1)
        out_alive[i:i + chunk] = alive_mass / (alive_mass + dead_mass)
        out_e[i:i + chunk] = (w_alive * gain).mean(axis=1) / (alive_mass + dead_mass)
    return out_e, out_alive


class BTYDModel:
    """BG/NBD + Gamma-Gamma fitted on a calibration snapshot."""

    def __init__(self, bgf_penalizer: float = 0.01, gg_penalizer: float = 0.01):
        self.bgf = BetaGeoFitter(penalizer_coef=bgf_penalizer)
        self.gg = GammaGammaFitter(penalizer_coef=gg_penalizer)
        self.repeat_mean_value = float("nan")
        self.freq_monetary_corr = float("nan")
        self.bgf_penalizer_used = bgf_penalizer

    def fit(self, summ: pd.DataFrame) -> BTYDModel:
        # lifetimes' optimiser occasionally fails to converge on small/young snapshots; retry with
        # a stronger L2 penalty (the penalty used is kept in `self.bgf_penalizer_used`).
        for pen in (self.bgf.penalizer_coef, 0.05, 0.2, 1.0):
            self.bgf = BetaGeoFitter(penalizer_coef=pen)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.bgf.fit(summ["frequency"], summ["recency"], summ["T"])
                self.bgf_penalizer_used = pen
                break
            except ConvergenceError:
                continue
        else:
            raise ConvergenceError("BG/NBD did not converge for any penalizer")
        repeat = summ[(summ["frequency"] > 0) & (summ["monetary_value"] > 0)]
        self.freq_monetary_corr = float(repeat["frequency"].corr(repeat["monetary_value"]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.gg.fit(repeat["frequency"], repeat["monetary_value"])
        self.repeat_mean_value = float(repeat["monetary_value"].mean())
        return self

    def predict(self, summ: pd.DataFrame, first_order_value: pd.Series,
                horizon_days: int = C.HORIZON_DAYS) -> pd.DataFrame:
        """Expected purchases, P(alive), expected order value and CLV for the next horizon."""
        exp_purchases, p_alive = posterior_purchases(
            dict(self.bgf.params_), summ["frequency"], summ["recency"], summ["T"], horizon_days)
        exp_purchases = pd.Series(exp_purchases, index=summ.index)
        p_alive = pd.Series(p_alive, index=summ.index)
        value = pd.Series(self.repeat_mean_value, index=summ.index, dtype=float)
        rep = (summ["frequency"] > 0) & (summ["monetary_value"] > 0)
        value[rep] = self.gg.conditional_expected_average_profit(
            summ.loc[rep, "frequency"], summ.loc[rep, "monetary_value"])
        # one-time buyers have no repeat monetary value: fall back to their own order value
        value[~rep] = first_order_value.reindex(summ.index[~rep]).fillna(self.repeat_mean_value)
        out = pd.DataFrame({
            "btyd_exp_purchases": exp_purchases, "btyd_p_alive": p_alive,
            "btyd_order_value": value,
        }, index=summ.index)
        out["btyd_clv"] = out["btyd_exp_purchases"] * out["btyd_order_value"]
        return out
