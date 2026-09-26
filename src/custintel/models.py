"""ML models (churn classifier, CLV regressor), tuning and metrics."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from . import config as C
from .features import BTYD_FEATURES, CALENDAR_FEATURES, FEATURES

LGBM_DEFAULT = {
    "n_estimators": 300, "learning_rate": 0.03, "num_leaves": 15, "min_child_samples": 40,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 5.0,
}


def _signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


def logistic_baseline() -> Pipeline:
    """Scaled logistic regression on signed-log features - the interpretable baseline."""
    return Pipeline([
        ("log", FunctionTransformer(_signed_log)),
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, C=0.5)),
    ])


def prep(X: pd.DataFrame, features: list[str] = FEATURES) -> pd.DataFrame:
    """Fill the few undefined features (one-off buyers have no gap / overdue ratio)."""
    X = X[features].copy()
    X["avg_gap_days"] = X["avg_gap_days"].fillna(X["tenure_days"].clip(lower=1))
    X["overdue_ratio"] = X["overdue_ratio"].fillna(0.0)
    return X.fillna(0.0).astype(float)


FEATURES_PLUS = [*FEATURES, *CALENDAR_FEATURES, *BTYD_FEATURES]


def churn_lgbm(params: dict | None = None) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(**(LGBM_DEFAULT | (params or {})), random_state=C.SEED, verbose=-1)


def clv_lgbm(params: dict | None = None) -> lgb.LGBMRegressor:
    """Tweedie objective: non-negative, zero-inflated, heavy-tailed revenue."""
    return lgb.LGBMRegressor(
        **(LGBM_DEFAULT | (params or {})), objective="tweedie", tweedie_variance_power=1.4,
        random_state=C.SEED, verbose=-1)


def search_space(rng: np.random.Generator) -> dict:
    return {
        "n_estimators": int(rng.choice([200, 300, 500])),
        "learning_rate": float(rng.choice([0.02, 0.03, 0.05])),
        "num_leaves": int(rng.choice([7, 15, 31])),
        "min_child_samples": int(rng.choice([20, 40, 80])),
        "feature_fraction": float(rng.choice([0.6, 0.8, 1.0])),
        "lambda_l2": float(rng.choice([1.0, 5.0, 20.0])),
    }


def tune(make_model, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, score, n_trials: int = 8,
         higher_is_better: bool = True, log=None) -> tuple[dict, float, list[dict]]:
    """Random search with customer-grouped 5-fold CV over the stacked training snapshots."""
    rng = np.random.default_rng(C.SEED)
    trials = [dict(LGBM_DEFAULT)] + [search_space(rng) for _ in range(n_trials)]
    cv = GroupKFold(n_splits=5)
    history, best, best_score = [], None, -np.inf
    for i, params in enumerate(trials):
        scores = []
        for tr, va in cv.split(X, y, groups):
            m = make_model(params).fit(X.iloc[tr], y[tr])
            scores.append(score(m, X.iloc[va], y[va]))
        mean = float(np.mean(scores))
        signed = mean if higher_is_better else -mean
        history.append({"trial": i, **params, "cv_score": mean, "cv_std": float(np.std(scores))})
        if log:
            log(i, params, mean, float(np.std(scores)))
        if signed > best_score:
            best, best_score = params, signed
    return best, abs(best_score), history


def _churn_score(m, X, y):
    return roc_auc_score(y, m.predict_proba(X)[:, 1])


def _clv_score(m, X, y):
    return float(np.abs(m.predict(X) - y).mean())


def churn_metrics(y, p) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    order = np.argsort(-p)
    k = max(1, int(len(y) * 0.1))
    top = y[order[:k]]
    return {
        "n": int(len(y)), "base_churn_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)) if 0 <= p.min() and p.max() <= 1 else None,
        "top_decile_churn_rate": float(top.mean()),
        "top_decile_lift": float(top.mean() / y.mean()),
    }


def clv_metrics(y, pred) -> dict:
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    order = np.argsort(-pred)
    k = max(1, int(len(y) * 0.1))
    return {
        "n": int(len(y)), "mae": float(np.abs(pred - y).mean()),
        "rmse": float(np.sqrt(((pred - y) ** 2).mean())),
        "spearman": float(spearmanr(pred, y).statistic),
        "total_actual": float(y.sum()), "total_predicted": float(pred.sum()),
        "top_decile_revenue_capture_pct": float(y[order[:k]].sum() / y.sum() * 100),
    }


def decile_table(y, pred, n: int = 10) -> pd.DataFrame:
    d = pd.DataFrame({"y": np.asarray(y, float), "pred": np.asarray(pred, float)})
    d["decile"] = pd.qcut(d["pred"].rank(method="first", ascending=False), n,
                          labels=range(1, n + 1))
    t = d.groupby("decile", observed=True).agg(customers=("y", "size"), predicted=("pred", "mean"),
                                               actual=("y", "mean"))
    return t.reset_index()
