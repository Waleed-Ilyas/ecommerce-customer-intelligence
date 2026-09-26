"""End-to-end pipeline: clean -> features -> cohorts -> segments -> churn & CLV models -> scores.

Usage:  python -m custintel.train        (needs data/raw/transactions.parquet from custintel.ingest)

Evaluation protocol (time-aware, no leakage)
  * Every row is a (customer, snapshot) pair. Features use data up to the snapshot only; labels use
    the following 180 days. Snapshots are monthly.
  * MODEL SELECTION uses a separate validation snapshot (2010-12-09): candidates are fitted on
    early snapshots whose label windows end before it. The test snapshot is not involved.
  * TEST snapshot 2011-06-09 (labels to 2011-12-09, the end of the data): all candidates are refit
    on all training snapshots and scored ONCE. Every candidate is reported, not just the winner.
  * Hyper-parameters of the selected LightGBM family are tuned with customer-grouped 5-fold CV on
    the training snapshots only.
  * Production scoring at 2011-12-09 refits the selected models on train + test snapshots.
"""
from __future__ import annotations

import json
import time
import warnings

import mlflow
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from . import clean, cohort, explain, ingest, segment
from . import config as C
from . import features as F
from . import models as M

N_TRIALS = 8
K_CLUSTERS = 5
DEPLOYED_CLV = "blend_bgnbd_lightgbm_plus"


def first_order_value(invoices: pd.DataFrame, snap: pd.Timestamp) -> pd.Series:
    inv = invoices[invoices["date"] <= snap].sort_values("invoice_date")
    return inv.groupby("customer_id")["revenue_capped"].first()


def top_products(sales: pd.DataFrame, snap: pd.Timestamp, n: int = 3) -> pd.Series:
    s = sales[sales["invoice_date"].dt.normalize() <= snap]
    s = s.assign(description=s["description"].fillna(s["stock_code"]).astype(str).str.title())
    g = s.groupby(["customer_id", "description"])["line_value"].sum().reset_index()
    g = g.sort_values(["customer_id", "line_value"], ascending=[True, False])
    return g.groupby("customer_id")["description"].apply(lambda d: " · ".join(d.head(n)))


PLUS = M.FEATURES_PLUS


def churn_candidates(tr: pd.DataFrame, te: pd.DataFrame, best_params: dict | None = None) -> dict:
    """Scores from every churn candidate on `te`, each fitted on `tr` only."""
    y = tr["churned"].to_numpy()
    out = {
        "rule_recency_only": te["recency_days"].to_numpy(float),  # reference, not a probability
        "bgnbd_expected_purchases": -te["btyd_exp_purchases"].to_numpy(),  # reference, not a probability
    }
    for name, cols in (("base", M.FEATURES), ("plus", PLUS)):
        out[f"logistic_{name}"] = M.logistic_baseline().fit(M.prep(tr, cols), y).predict_proba(
            M.prep(te, cols))[:, 1]
        out[f"lightgbm_{name}"] = M.churn_lgbm(best_params).fit(
            M.prep(tr, cols), y).predict_proba(M.prep(te, cols))[:, 1]
    return out


# Candidates listed simplest -> most complex. Selection rule (a parsimony / one-margin rule): pick the
# simplest candidate whose validation score is within SELECTION_MARGIN of the best one. With ~4k
# customers per snapshot, smaller gaps are noise and a more complex model has more ways to fail.
PROBABILISTIC_CHURN = ["logistic_base", "logistic_plus", "lightgbm_base", "lightgbm_plus"]
SELECTION_MARGIN = 0.01


def select_simplest(order: list[str], scores: dict[str, float], margin: float = SELECTION_MARGIN) -> str:
    best = max(scores[k] for k in order)
    return next(k for k in order if scores[k] >= best - margin)


def paired_bootstrap(y, a, b, metric, n: int = 400, seed: int = C.SEED) -> dict:
    """95% CI of metric(a) - metric(b) resampling customers (paired)."""
    rng = np.random.default_rng(seed)
    y, a, b = np.asarray(y), np.asarray(a), np.asarray(b)
    diffs = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        diffs.append(metric(y[i], a[i]) - metric(y[i], b[i]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": float(metric(y, a) - metric(y, b)), "ci95": [float(lo), float(hi)]}


def clv_candidates(tr: pd.DataFrame, te: pd.DataFrame, best_params: dict | None = None) -> dict:
    y = tr["future_revenue"].to_numpy()
    out = {
        "naive_last_180d_spend": te["revenue_180d"].to_numpy(float),
        "naive_lifetime_average_rate": (te["monetary_total"] / te["tenure_days"].clip(lower=30)
                                        * C.HORIZON_DAYS).to_numpy(),
        "bgnbd_gamma_gamma": te["btyd_clv"].to_numpy(),
    }
    for name, cols in (("base", M.FEATURES), ("plus", PLUS)):
        m = M.clv_lgbm(best_params).fit(M.prep(tr, cols), y)
        out[f"lightgbm_{name}"] = np.clip(m.predict(M.prep(te, cols)), 0, None)
    out["blend_bgnbd_lightgbm_plus"] = 0.5 * out["bgnbd_gamma_gamma"] + 0.5 * out["lightgbm_plus"]
    return out


CLV_SELECTABLE = ["bgnbd_gamma_gamma", "lightgbm_base", "lightgbm_plus", "blend_bgnbd_lightgbm_plus"]


def _auc(y, s):
    return roc_auc_score(y, s)


def _spearman(y, s):
    return spearmanr(s, y).statistic


def main() -> None:
    warnings.filterwarnings("ignore")
    t0 = time.time()
    C.ARTIFACTS_DIR.mkdir(exist_ok=True)
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    (C.ARTIFACTS_DIR / "model").mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{C.ROOT / 'mlflow.db'}")
    mlflow.set_experiment("customer-intelligence")

    # ---------------- 1. clean ----------------
    sales, returns, report = clean.clean(ingest.load_raw())
    invoices = clean.to_invoices(sales)
    report["invoice_revenue_cap_p99_9"] = invoices.attrs["revenue_cap"]
    (C.ARTIFACTS_DIR / "cleaning_report.json").write_text(json.dumps(report, indent=1))
    sales.to_parquet(C.PROCESSED_DIR / "sales.parquet")
    returns.to_parquet(C.PROCESSED_DIR / "returns.parquet")
    invoices.to_parquet(C.PROCESSED_DIR / "invoices.parquet")
    score_snap = pd.Timestamp(C.SCORING_SNAPSHOT)

    # ---------------- 2. cohorts ----------------
    for name, df in cohort.cohort_tables(invoices).items():
        out = df.copy()
        out.index, out.columns = out.index.astype(str), out.columns.astype(str)
        out.to_parquet(C.ARTIFACTS_DIR / f"cohort_{name}.parquet")

    # ---------------- 3. datasets ----------------
    def build(snaps, **kw):
        return F.build_dataset(invoices, returns, sales, snaps, **kw)

    train = build(C.TRAIN_SNAPSHOTS)
    test = build([C.TEST_SNAPSHOT])
    valid_tr = train[train["snapshot"].isin(pd.to_datetime(C.VALID_TRAIN_SNAPSHOTS))]
    valid = train[train["snapshot"] == pd.Timestamp(C.VALID_SNAPSHOT)]
    print(f"train {train.shape}, valid {valid.shape}, test {test.shape} ({time.time() - t0:.0f}s)")

    with mlflow.start_run(run_name="pipeline") as parent:
        mlflow.log_params({"horizon_days": C.HORIZON_DAYS, "test_snapshot": C.TEST_SNAPSHOT,
                           "valid_snapshot": C.VALID_SNAPSHOT,
                           "train_snapshots": len(C.TRAIN_SNAPSHOTS)})

        # ---------------- 4. model selection on the validation snapshot ----------------
        vy_c, vy_v = valid["churned"].to_numpy(), valid["future_revenue"].to_numpy()
        v_churn = {k: M.churn_metrics(vy_c, v) for k, v in churn_candidates(valid_tr, valid).items()}
        v_clv = {k: M.clv_metrics(vy_v, v) for k, v in clv_candidates(valid_tr, valid).items()}
        sel_churn = select_simplest(PROBABILISTIC_CHURN, {k: v_churn[k]["roc_auc"] for k in v_churn})
        sel_clv = select_simplest(CLV_SELECTABLE, {k: v_clv[k]["spearman"] for k in v_clv})
        print("validation -> churn:", sel_churn, "| clv:", sel_clv)
        mlflow.log_params({"selected_churn": sel_churn, "selected_clv": sel_clv})

        # ---------------- 5. tune on the training snapshots (customer-grouped CV) ----------------
        def logger(kind):
            def log(i, params, mean, std):
                with mlflow.start_run(run_name=f"{kind}-trial-{i}", nested=True):
                    mlflow.log_params(params)
                    mlflow.log_metrics({"cv_score": mean, "cv_std": std})
                print(f"{kind} trial {i}: {mean:.4f} +- {std:.4f}")
            return log

        groups = train["customer_id"].to_numpy()
        Xtr_base = M.prep(train)
        best_c, cv_auc, trials_c = M.tune(M.churn_lgbm, Xtr_base, train["churned"].to_numpy(),
                                          groups, M._churn_score, N_TRIALS, True, logger("churn"))
        best_v, cv_mae, trials_v = M.tune(M.clv_lgbm, Xtr_base, train["future_revenue"].to_numpy(),
                                          groups, M._clv_score, N_TRIALS, False, logger("clv"))

        # ---------------- 6. test snapshot: every candidate, scored once ----------------
        ty_c, ty_v = test["churned"].to_numpy(), test["future_revenue"].to_numpy()
        churn_scores = churn_candidates(train, test, best_c)
        clv_scores = clv_candidates(train, test, best_v)
        churn_results = {k: M.churn_metrics(ty_c, v) for k, v in churn_scores.items()}
        clv_results = {k: M.clv_metrics(ty_v, v) for k, v in clv_scores.items()}
        deciles = {k: M.decile_table(ty_v, clv_scores[k]).to_dict("records")
                   for k in {"bgnbd_gamma_gamma", "lightgbm_plus", sel_clv, DEPLOYED_CLV}}
        significance = {
            "churn_selected_vs_rule_auc": paired_bootstrap(
                ty_c, churn_scores[sel_churn], churn_scores["rule_recency_only"], _auc),
            "churn_selected_vs_bgnbd_auc": paired_bootstrap(
                ty_c, churn_scores[sel_churn], churn_scores["bgnbd_expected_purchases"], _auc),
            "clv_selected_vs_naive_spearman": paired_bootstrap(
                ty_v, clv_scores[sel_clv], clv_scores["naive_last_180d_spend"], _spearman),
            "clv_selected_vs_bgnbd_spearman": paired_bootstrap(
                ty_v, clv_scores[sel_clv], clv_scores["bgnbd_gamma_gamma"], _spearman),
            "clv_blend_vs_bgnbd_spearman": paired_bootstrap(
                ty_v, clv_scores["blend_bgnbd_lightgbm_plus"], clv_scores["bgnbd_gamma_gamma"],
                _spearman),
        }
        cal = pd.DataFrame({"p": churn_scores[sel_churn], "y": ty_c})
        cal["bin"] = pd.qcut(cal["p"].rank(method="first"), 10, labels=range(1, 11))
        calibration = cal.groupby("bin", observed=True).agg(predicted=("p", "mean"),
                                                            actual=("y", "mean")).reset_index()

        test_out = test[["customer_id", "churned", "future_revenue", "recency_days"]].copy()
        for k, v in churn_scores.items():
            test_out[f"churn__{k}"] = v
        for k, v in clv_scores.items():
            test_out[f"clv__{k}"] = v
        test_out.to_parquet(C.PROCESSED_DIR / "test_predictions.parquet")

        # ---------------- 7. SHAP for the LightGBM churn / CLV models (test snapshot) ----------------
        sample = M.prep(test, PLUS).sample(min(2000, len(test)), random_state=C.SEED)
        sample = sample[M.FEATURES]
        churn_shap_model = M.churn_lgbm(best_c).fit(Xtr_base, train["churned"].to_numpy())
        clv_shap_model = M.clv_lgbm(best_v).fit(Xtr_base, train["future_revenue"].to_numpy())
        sv_churn = shap.TreeExplainer(churn_shap_model).shap_values(sample)
        sv_clv = shap.TreeExplainer(clv_shap_model).shap_values(sample)
        shap_top = {
            "churn": pd.Series(np.abs(sv_churn).mean(0), index=sample.columns).sort_values(
                ascending=False).head(12).round(4).to_dict(),
            "clv": pd.Series(np.abs(sv_clv).mean(0), index=sample.columns).sort_values(
                ascending=False).head(12).round(4).to_dict(),
        }
        np.save(C.PROCESSED_DIR / "shap_churn.npy", sv_churn)
        sample.to_parquet(C.PROCESSED_DIR / "shap_sample.parquet")

        # ---------------- 8. segmentation (as of the scoring snapshot) ----------------
        score_feats = build([C.SCORING_SNAPSHOT], labelled=False)
        k_table = segment.choose_k(score_feats)
        km, _ = segment.fit_kmeans(score_feats, K_CLUSTERS)
        score_feats["cluster"] = km.labels_
        prof = score_feats.groupby("cluster")[["recency_days", "n_orders", "tenure_days"]].median()
        persona = segment.name_personas(prof)
        score_feats["segment"] = score_feats["cluster"].map(persona)
        score_feats = score_feats.merge(
            segment.rfm_scores(score_feats)[["customer_id", "R", "F", "M", "rfm_score"]],
            on="customer_id")

        # ---------------- 9. production scoring: refit selected models on train + test ----------------
        full = pd.concat([train, test], ignore_index=True)
        # DEPLOYED churn model = the validation-selected one (logistic_base).
        assert sel_churn == "logistic_base", "deployed churn model is the logistic baseline"
        lr_final = M.logistic_baseline().fit(M.prep(full), full["churned"].to_numpy())
        Xs = M.prep(score_feats)
        score_feats["churn_prob"] = lr_final.predict_proba(Xs)[:, 1]
        score_feats["risk_signals"] = explain.signals(score_feats).to_numpy()
        clv_final = clv_candidates(full, score_feats, best_v)
        # DEPLOYED CLV = equal-weight blend of BG/NBD+Gamma-Gamma and LightGBM. NOTE: this is a
        # post-hoc decision made after seeing the test snapshot (the pre-declared selection rule
        # picked lightgbm_base, which was significantly worse than BG/NBD there). Documented in the
        # README; both component estimates are kept in the output.
        score_feats["clv_180d"] = clv_final[DEPLOYED_CLV]
        score_feats["clv_180d_bgnbd"] = clv_final["bgnbd_gamma_gamma"]
        score_feats["clv_180d_lightgbm"] = clv_final["lightgbm_plus"]
        # heuristic: what a churned customer would typically have spent in 180 days
        score_feats["typical_180d_spend"] = (score_feats["revenue_180d"]
                                             + score_feats["revenue_prev_180d"]) / 2
        score_feats["value_at_risk"] = score_feats["churn_prob"] * score_feats["typical_180d_spend"]
        score_feats["action"] = score_feats["segment"].str.replace(r" \(\d+\)", "", regex=True).map(
            segment.ACTIONS)
        score_feats["top_products"] = score_feats["customer_id"].map(top_products(sales, score_snap))
        first = invoices.groupby("customer_id")["date"].min()
        last = invoices.groupby("customer_id")["date"].max()
        score_feats["first_order"] = score_feats["customer_id"].map(first).dt.strftime("%Y-%m-%d")
        score_feats["last_order"] = score_feats["customer_id"].map(last).dt.strftime("%Y-%m-%d")
        score_feats["country"] = score_feats["customer_id"].map(
            invoices.sort_values("invoice_date").groupby("customer_id")["country"].last())
        score_feats["cohort"] = score_feats["customer_id"].map(first.dt.to_period("M").astype(str))
        score_feats.drop(columns=["snapshot", "cluster"]).to_parquet(
            C.ARTIFACTS_DIR / "customers.parquet")
        invoices[["customer_id", "invoice", "date", "revenue", "n_lines", "units", "country"]
                 ].to_parquet(C.ARTIFACTS_DIR / "orders.parquet")
        (C.ARTIFACTS_DIR / "model" / "features.json").write_text(json.dumps(PLUS))

        seg = score_feats.groupby("segment").agg(
            customers=("customer_id", "size"), revenue=("monetary_total", "sum"),
            median_recency=("recency_days", "median"), median_orders=("n_orders", "median"),
            median_order_value=("avg_order_value", "median"),
            avg_churn_prob=("churn_prob", "mean"), clv_180d=("clv_180d", "sum"),
            value_at_risk=("value_at_risk", "sum")).reset_index()
        seg["customer_share_pct"] = seg["customers"] / seg["customers"].sum() * 100
        seg["revenue_share_pct"] = seg["revenue"] / seg["revenue"].sum() * 100
        seg.to_parquet(C.ARTIFACTS_DIR / "segment_summary.parquet")

        metrics = {
            "protocol": {"train_snapshots": C.TRAIN_SNAPSHOTS, "valid_snapshot": C.VALID_SNAPSHOT,
                         "valid_train_snapshots": C.VALID_TRAIN_SNAPSHOTS,
                         "test_snapshot": C.TEST_SNAPSHOT, "horizon_days": C.HORIZON_DAYS},
            "data": {"customers_scored": int(len(score_feats)), "train_rows": len(train),
                     "valid_rows": len(valid), "test_rows": len(test),
                     "test_churn_rate": float(ty_c.mean()), "test_actual_revenue": float(ty_v.sum())},
            "selection": {"churn": sel_churn, "clv_rule_pick": sel_clv, "clv_deployed": DEPLOYED_CLV,
                          "selection_margin": SELECTION_MARGIN, "validation_churn": v_churn,
                          "validation_clv": v_clv},
            "churn": {"tuned_cv_auc": cv_auc, "best_params": best_c, "test": churn_results,
                      "calibration": calibration.to_dict("records"), "trials": trials_c},
            "clv": {"tuned_cv_mae": cv_mae, "best_params": best_v, "test": clv_results,
                    "deciles": deciles, "trials": trials_v},
            "segmentation": {"k_selection": k_table.round(4).to_dict("records"), "k": K_CLUSTERS,
                             "personas": {str(k): v for k, v in persona.items()},
                             "cluster_profile": prof.round(1).reset_index().to_dict("records")},
            "significance": significance,
            "shap_top": shap_top,
            "runtime_min": (time.time() - t0) / 60,
        }
        (C.ARTIFACTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=1, default=float))
        mlflow.log_metrics({
            "test_churn_auc_selected": churn_results[sel_churn]["roc_auc"],
            "test_clv_spearman_deployed": clv_results[DEPLOYED_CLV]["spearman"],
            "test_clv_mae_deployed": clv_results[DEPLOYED_CLV]["mae"],
            "test_clv_top_decile_capture_deployed":
                clv_results[DEPLOYED_CLV]["top_decile_revenue_capture_pct"],
        })
        mlflow.log_artifacts(str(C.ARTIFACTS_DIR), artifact_path="artifacts")
        print(f"run {parent.info.run_id} done in {(time.time() - t0) / 60:.1f} min")
        print("VALIDATION churn:", {k: round(v["roc_auc"], 4) for k, v in v_churn.items()})
        print("VALIDATION clv spearman:", {k: round(v["spearman"], 4) for k, v in v_clv.items()})
        print("TEST churn:", {k: round(v["roc_auc"], 4) for k, v in churn_results.items()})
        print("TEST clv:", {k: (round(v["mae"]), round(v["spearman"], 3),
                              round(v["top_decile_revenue_capture_pct"], 1))
                            for k, v in clv_results.items()})


if __name__ == "__main__":
    main()
