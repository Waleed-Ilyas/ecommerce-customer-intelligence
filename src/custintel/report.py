"""Generate report figures from a finished training run.  Usage: python -m custintel.report"""
from __future__ import annotations

import json

import pandas as pd

from . import config as C
from . import viz


def main() -> None:
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    A, F = C.ARTIFACTS_DIR, C.FIGURES_DIR
    metrics = json.loads((A / "metrics.json").read_text())
    report = json.loads((A / "cleaning_report.json").read_text())
    customers = pd.read_parquet(A / "customers.parquet")
    summary = pd.read_parquet(A / "segment_summary.parquet")
    pred = pd.read_parquet(C.PROCESSED_DIR / "test_predictions.parquet")

    viz.plot_cleaning_waterfall(report, F / "01_cleaning_waterfall.png")
    viz.plot_cohorts(pd.read_parquet(A / "cohort_retention_pct.parquet"),
                     pd.read_parquet(A / "cohort_cohort_size.parquet")["customers"],
                     F / "02_cohort_retention.png")
    viz.plot_k_selection(pd.DataFrame(metrics["segmentation"]["k_selection"]),
                         metrics["segmentation"]["k"], F / "03_k_selection.png")
    viz.plot_segments(customers, summary, F / "04_segments.png")
    viz.plot_churn_roc(pred, F / "05_churn_roc.png", {
        "churn__rule_recency_only": ("Rule: days since last order", viz.BRONZE),
        "churn__bgnbd_expected_purchases": ("BG/NBD", "#B0A99F"),
        "churn__logistic_base": ("Logistic regression (deployed)", viz.GOLD),
        "churn__lightgbm_base": ("LightGBM", viz.CYAN),
    })
    viz.plot_clv_gains(pred, F / "06_clv_gains.png", {
        "clv__naive_last_180d_spend": ("Naive: last 6-month spend", viz.BRONZE),
        "clv__lightgbm_base": ("LightGBM", viz.CYAN),
        "clv__bgnbd_gamma_gamma": ("BG/NBD + Gamma-Gamma", "#B0A99F"),
        "clv__blend_bgnbd_lightgbm_plus": ("Blend (deployed)", viz.GOLD),
    })
    viz.plot_shap_bars(metrics["shap_top"]["churn"], "Churn: what LightGBM relies on (SHAP)",
                       F / "07_shap_churn.png")
    viz.plot_shap_bars(metrics["shap_top"]["clv"], "6-month revenue: what LightGBM relies on (SHAP)",
                       F / "08_shap_clv.png")
    print("figures written to", F)


if __name__ == "__main__":
    main()
