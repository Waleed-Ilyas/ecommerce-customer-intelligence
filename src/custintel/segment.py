"""RFM scoring and K-Means segmentation with persona naming."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from . import config as C

RFM_COLS = ["recency_days", "n_orders", "monetary_total"]

ACTIONS = {
    "Champions": "Reward and retain: early access, VIP tier, referral programme. Do not discount.",
    "Loyal": "Grow basket size: bundles, cross-sell, loyalty points toward the next tier.",
    "Potential Loyalists": "Convert to regulars: second-order incentive, replenishment reminders.",
    "New Customers": "Onboard: welcome series, best-seller recommendations, nudge a 2nd purchase.",
    "At Risk": "Win back now: personalised 'we miss you' offer, time-limited, best past categories.",
    "Hibernating": "Low-cost reactivation: seasonal email, then suppress to save spend.",
    "Lost": "Last-chance survey/offer; otherwise remove from paid campaigns.",
}


def rfm_scores(feats: pd.DataFrame) -> pd.DataFrame:
    """Classic 1-5 quintile RFM scores (5 = best)."""
    out = feats[["customer_id", *RFM_COLS]].copy()
    out["R"] = pd.qcut(out["recency_days"].rank(method="first", ascending=False), 5,
                       labels=[1, 2, 3, 4, 5]).astype(int)
    out["F"] = pd.qcut(out["n_orders"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    out["M"] = pd.qcut(out["monetary_total"].rank(method="first"), 5,
                       labels=[1, 2, 3, 4, 5]).astype(int)
    out["rfm_score"] = out["R"] * 100 + out["F"] * 10 + out["M"]
    return out


def _matrix(feats: pd.DataFrame) -> np.ndarray:
    return np.log1p(feats[RFM_COLS].to_numpy(float))


def choose_k(feats: pd.DataFrame, k_range=range(2, 10), sample: int = 4000) -> pd.DataFrame:
    """Inertia (elbow) and silhouette for each k."""
    X = StandardScaler().fit_transform(_matrix(feats))
    rng = np.random.default_rng(C.SEED)
    idx = rng.choice(len(X), size=min(sample, len(X)), replace=False)
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=C.SEED).fit(X)
        rows.append({"k": k, "inertia": float(km.inertia_),
                     "silhouette": float(silhouette_score(X[idx], km.labels_[idx]))})
    return pd.DataFrame(rows)


def fit_kmeans(feats: pd.DataFrame, k: int) -> tuple[KMeans, StandardScaler]:
    scaler = StandardScaler().fit(_matrix(feats))
    km = KMeans(n_clusters=k, n_init=20, random_state=C.SEED).fit(scaler.transform(_matrix(feats)))
    return km, scaler


def name_personas(profile: pd.DataFrame) -> dict[int, str]:
    """Persona names from cluster *medians* using explicit, documented thresholds.

    `profile` is indexed by cluster id with columns recency_days, n_orders, tenure_days.
      recent (<= 60 days since last order):  >= 15 orders Champions | >= 5 Loyal |
                                             tenure <= 200 days New Customers | else Potential Loyalists
      60-330 days:                           >= 3 orders At Risk | else Hibernating
      > 330 days:                            Lost
    """
    names: dict[int, str] = {}
    for c, row in profile.iterrows():
        rec, freq, tenure = row["recency_days"], row["n_orders"], row["tenure_days"]
        if rec <= 60:
            if freq >= 15:
                names[c] = "Champions"
            elif freq >= 5:
                names[c] = "Loyal"
            elif tenure <= 200:
                names[c] = "New Customers"
            else:
                names[c] = "Potential Loyalists"
        elif rec <= 330:
            names[c] = "At Risk" if freq >= 3 else "Hibernating"
        else:
            names[c] = "Lost"
    # guarantee unique labels if two clusters fall into the same rule
    seen: dict[str, int] = {}
    for c in sorted(names):
        base = names[c]
        seen[base] = seen.get(base, 0) + 1
        if seen[base] > 1:
            names[c] = f"{base} ({seen[base]})"
    return names
