"""E-commerce Customer Intelligence - Streamlit app (segments, cohorts, CLV, churn risk)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
GOLD, IVORY, BG, PANEL, CYAN, BRONZE = "#D9B26A", "#F2EDE4", "#0A0A0C", "#15110F", "#4FD1C5", "#8B5E3C"
SEG_COLORS = {"Champions": GOLD, "Loyal": CYAN, "New Customers": "#9ad1a0", "At Risk": "#ff9f5a",
              "Lost": "#7d7368", "Hibernating": "#a58b6f", "Potential Loyalists": "#c9b8e8"}

st.set_page_config(page_title="Customer Intelligence", page_icon="🛍️", layout="wide")
st.markdown(
    f"""
    <style>
    .block-container {{padding-top: 2rem; max-width: 1250px;}}
    h1, h2, h3 {{font-family: Georgia, 'Times New Roman', serif; letter-spacing: -0.01em;}}
    h1 {{font-weight: 400; font-size: 2.6rem;}}
    .kpi {{background: {PANEL}; border: 1px solid rgba(242,237,228,.08); border-radius: 12px;
          padding: 1rem 1.2rem; height: 100%;}}
    .kpi .label {{font-size: .72rem; letter-spacing: .1em; text-transform: uppercase; opacity: .6;}}
    .kpi .value {{font-size: 1.9rem; font-family: Georgia, serif; line-height: 1.25;}}
    .kpi .sub {{font-size: .85rem; opacity: .75;}}
    .pill {{display:inline-block; padding:.15rem .7rem; border-radius:99px; font-size:.8rem;
           border:1px solid rgba(242,237,228,.2); margin-right:.4rem;}}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load():
    customers = pd.read_parquet(ART / "customers.parquet")
    customers["segment"] = customers["segment"].astype(str)
    return {
        "customers": customers,
        "orders": pd.read_parquet(ART / "orders.parquet"),
        "segments": pd.read_parquet(ART / "segment_summary.parquet"),
        "retention": pd.read_parquet(ART / "cohort_retention_pct.parquet"),
        "cohort_size": pd.read_parquet(ART / "cohort_cohort_size.parquet"),
        "metrics": json.loads((ART / "metrics.json").read_text()),
        "cleaning": json.loads((ART / "cleaning_report.json").read_text()),
    }


D = load()
CUST, SEGS, M = D["customers"], D["segments"], D["metrics"]


def gbp(x: float) -> str:
    return f"£{x:,.0f}" if abs(x) < 1e6 else f"£{x / 1e6:,.2f}M"


def kpi(label: str, value: str, sub: str = "") -> str:
    return (f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div>'
            f'<div class="sub">{sub}</div></div>')


def style(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=IVORY), margin=dict(l=10, r=10, t=40, b=10))
    fig.update_xaxes(gridcolor="rgba(242,237,228,.06)")
    fig.update_yaxes(gridcolor="rgba(242,237,228,.06)")
    return fig


st.title("Customer Intelligence")
st.caption("Segmentation · cohorts · 6-month CLV · churn risk — UCI Online Retail II "
           "(real UK gift-retailer transactions, Dec 2009 - Dec 2011). Scores as of 2011-12-09.")

tab_over, tab_seg, tab_coh, tab_cust, tab_camp, tab_model = st.tabs(
    ["Overview", "Segment explorer", "Cohort retention", "Customer lookup", "Campaign lists",
     "Method & accuracy"])

# ------------------------------------------------------------------ overview
with tab_over:
    champ = SEGS[SEGS.segment == "Champions"].iloc[0]
    atrisk = SEGS[SEGS.segment == "At Risk"].iloc[0]
    c = st.columns(4)
    c[0].markdown(kpi("Customers scored", f"{len(CUST):,}", "with ≥ 1 purchase"), unsafe_allow_html=True)
    c[1].markdown(kpi("Revenue in data", gbp(CUST.monetary_total.sum()), "identified customers, large orders capped"),
                  unsafe_allow_html=True)
    c[2].markdown(kpi("Champions", f"{champ.customer_share_pct:.1f}%",
                      f"of customers · {champ.revenue_share_pct:.0f}% of revenue"), unsafe_allow_html=True)
    c[3].markdown(kpi("Expected 6-month revenue", gbp(CUST.clv_180d.sum()), "model-based (CLV)"),
                  unsafe_allow_html=True)
    st.markdown("")
    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("#### Where the revenue comes from")
        fig = px.treemap(SEGS, path=["segment"], values="revenue", color="segment",
                         color_discrete_map=SEG_COLORS)
        fig.update_traces(texttemplate="<b>%{label}</b><br>%{percentRoot:.0%}", textfont_size=15)
        st.plotly_chart(style(fig, 380), width="stretch")
    with right:
        st.markdown("#### What to do first")
        st.markdown(
            f"- **Protect {champ.customers:,.0f} Champions** - they bring {champ.revenue_share_pct:.0f}% "
            f"of revenue; churn risk averages just {champ.avg_churn_prob * 100:.0f}%.\n"
            f"- **Win back At-Risk customers** - {atrisk.customers:,.0f} buyers with a purchase "
            f"history, average churn risk {atrisk.avg_churn_prob * 100:.0f}%, "
            f"{gbp(atrisk.value_at_risk)} of estimated value at risk.\n"
            f"- **Convert New customers** - {SEGS[SEGS.segment == 'New Customers'].customers.iloc[0]:,.0f} "
            f"recent first-time buyers; the second order is the key retention lever.\n"
            "- **Stop spending on Lost customers** - 28% of the base, under 3% of revenue.")
        st.caption("Value at risk = churn probability × typical 6-month spend (average of the last two "
                   "6-month periods). A heuristic ranking, not a forecast of lost revenue.")

# ------------------------------------------------------------------ segments
with tab_seg:
    order = SEGS.sort_values("revenue", ascending=False)["segment"].tolist()
    seg = st.radio("Segment", order, horizontal=True)
    row = SEGS[SEGS.segment == seg].iloc[0]
    sub = CUST[CUST.segment == seg]
    c = st.columns(5)
    c[0].markdown(kpi("Customers", f"{row.customers:,.0f}", f"{row.customer_share_pct:.1f}% of base"),
                  unsafe_allow_html=True)
    c[1].markdown(kpi("Revenue share", f"{row.revenue_share_pct:.1f}%", gbp(row.revenue)),
                  unsafe_allow_html=True)
    c[2].markdown(kpi("Median recency", f"{row.median_recency:.0f} days", "since last order"),
                  unsafe_allow_html=True)
    c[3].markdown(kpi("Median orders", f"{row.median_orders:.0f}", f"AOV £{row.median_order_value:,.0f}"),
                  unsafe_allow_html=True)
    c[4].markdown(kpi("Avg churn risk", f"{row.avg_churn_prob * 100:.0f}%", "next 6 months"),
                  unsafe_allow_html=True)
    st.info(f"**Recommended action:** {sub.action.iloc[0]}", icon="🎯")
    fig = px.scatter(CUST, x="recency_days", y="monetary_total", color="segment", log_y=True,
                     color_discrete_map=SEG_COLORS, opacity=0.55,
                     labels={"recency_days": "days since last order", "monetary_total": "lifetime spend (£, log)"})
    fig.update_traces(marker=dict(size=5))
    fig.for_each_trace(lambda t: t.update(opacity=0.9 if t.name == seg else 0.12))
    fig.update_layout(showlegend=False, title=f"{seg} highlighted (other segments dimmed)")
    st.plotly_chart(style(fig, 430), width="stretch")
    st.markdown(f"#### Top 15 {seg} customers by expected 6-month revenue")
    st.dataframe(sub.sort_values("clv_180d", ascending=False).head(15)[
        ["customer_id", "country", "n_orders", "monetary_total", "recency_days", "churn_prob",
         "clv_180d", "top_products"]], hide_index=True, width="stretch",
        column_config={"monetary_total": st.column_config.NumberColumn("lifetime £", format="£%.0f"),
                       "clv_180d": st.column_config.NumberColumn("6-mo CLV £", format="£%.0f"),
                       "churn_prob": st.column_config.ProgressColumn("churn risk", min_value=0,
                                                                     max_value=1, format="%.2f")})

# ------------------------------------------------------------------ cohorts
with tab_coh:
    ret = D["retention"].copy()
    ret = ret[[c for c in ret.columns if int(c) <= 12]]
    sizes = D["cohort_size"]["customers"]
    st.markdown("#### Of every 100 new customers, how many order again in month N?")
    z = ret.drop(columns=["0"]).values
    fig = go.Figure(go.Heatmap(
        z=z, x=[f"M{c}" for c in ret.columns[1:]], y=[f"{i} (n={int(sizes[i])})" for i in ret.index],
        colorscale=[[0, "#15110F"], [0.5, "#8B5E3C"], [1, "#D9B26A"]], zmin=0, zmax=45,
        colorbar=dict(title="% active"), hovertemplate="%{y}<br>%{x}: %{z:.0f}%<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(style(fig, 620), width="stretch")
    curve = ret.drop(columns=["0"]).mean()
    st.write(f"**Average cohort:** {curve.iloc[0]:.0f}% order again in month 1, "
             f"{curve.iloc[5]:.0f}% in month 6 and {curve.iloc[-1]:.0f}% in month 12. "
             "The December-2009 cohort (the data starts that month, so it includes the retailer's existing "
             "customers) is far stickier - 33-50% order every month - than any later cohort.")
    st.caption("Cells are the share of a cohort's customers who placed at least one order in that calendar "
               "month; later cohorts have fewer observable months (Dec 2011 data ends on the 9th).")

# ------------------------------------------------------------------ lookup
with tab_cust:
    ids = CUST.sort_values("clv_180d", ascending=False)["customer_id"].tolist()
    cid = st.selectbox("Customer ID (sorted by expected 6-month revenue)", ids, index=0)
    r = CUST[CUST.customer_id == cid].iloc[0]
    colr = SEG_COLORS.get(r.segment.split(" (")[0], IVORY)
    st.markdown(f'<span class="pill" style="border-color:{colr};color:{colr}">{r.segment}</span>'
                f'<span class="pill">{r.country}</span><span class="pill">first order {r.first_order}</span>'
                f'<span class="pill">last order {r.last_order}</span>', unsafe_allow_html=True)
    st.markdown("")
    c = st.columns(4)
    c[0].markdown(kpi("Churn risk", f"{r.churn_prob * 100:.0f}%", "no order in next 180 days"),
                  unsafe_allow_html=True)
    c[1].markdown(kpi("Expected 6-month revenue", gbp(r.clv_180d),
                      f"BG/NBD {gbp(r.clv_180d_bgnbd)} · LightGBM {gbp(r.clv_180d_lightgbm)}"),
                  unsafe_allow_html=True)
    c[2].markdown(kpi("Value at risk", gbp(r.value_at_risk), "churn prob × typical spend"),
                  unsafe_allow_html=True)
    c[3].markdown(kpi("Lifetime", gbp(r.monetary_total), f"{r.n_orders:.0f} orders · AOV {gbp(r.avg_order_value)}"),
                  unsafe_allow_html=True)
    st.markdown("")
    st.markdown(f"**Signals:** {r.risk_signals}")
    st.markdown(f"**Favourite products:** {r.top_products}")
    st.info(f"**Recommended action:** {r.action}", icon="🎯")
    hist = D["orders"][D["orders"].customer_id == cid].sort_values("date")
    fig = go.Figure(go.Bar(x=hist["date"], y=hist["revenue"], marker_color=GOLD,
                           hovertemplate="%{x|%d %b %Y}: £%{y:,.0f}<extra></extra>"))
    fig.update_yaxes(title="order value (£)")
    st.plotly_chart(style(fig.update_layout(title="Order history"), 300), width="stretch")

# ------------------------------------------------------------------ campaigns
with tab_camp:
    st.markdown("#### Build a target list")
    presets = {
        "Win-back: valuable and slipping": dict(segs=["At Risk", "Loyal"], risk=0.5, var=100),
        "Protect: Champions showing risk": dict(segs=["Champions"], risk=0.25, var=0),
        "Nurture: new customers likely to lapse": dict(segs=["New Customers"], risk=0.5, var=0),
        "Custom": None,
    }
    choice = st.selectbox("Preset", list(presets))
    p = presets[choice] or dict(segs=sorted(CUST.segment.unique()), risk=0.0, var=0)
    c1, c2, c3 = st.columns(3)
    segs = c1.multiselect("Segments", sorted(CUST.segment.unique()), default=p["segs"])
    min_risk = c2.slider("Min churn risk", 0.0, 1.0, float(p["risk"]), 0.05)
    min_var = c3.number_input("Min value at risk (£)", 0, 20000, int(p["var"]), 25)
    lst = CUST[CUST.segment.isin(segs) & (CUST.churn_prob >= min_risk) & (CUST.value_at_risk >= min_var)]
    lst = lst.sort_values("value_at_risk", ascending=False)
    c = st.columns(3)
    c[0].markdown(kpi("Customers in list", f"{len(lst):,}", f"{len(lst) / len(CUST) * 100:.1f}% of base"),
                  unsafe_allow_html=True)
    c[1].markdown(kpi("Value at risk", gbp(lst.value_at_risk.sum()), "heuristic, see Overview"),
                  unsafe_allow_html=True)
    c[2].markdown(kpi("Historical revenue", gbp(lst.monetary_total.sum()),
                      f"{lst.monetary_total.sum() / CUST.monetary_total.sum() * 100:.0f}% of all revenue"),
                  unsafe_allow_html=True)
    cols = ["customer_id", "segment", "country", "churn_prob", "clv_180d", "value_at_risk",
            "recency_days", "n_orders", "monetary_total", "action", "risk_signals"]
    st.dataframe(lst[cols].head(500), hide_index=True, width="stretch",
                 column_config={"churn_prob": st.column_config.ProgressColumn("churn risk", min_value=0,
                                                                              max_value=1, format="%.2f"),
                                "clv_180d": st.column_config.NumberColumn("6-mo CLV £", format="£%.0f"),
                                "value_at_risk": st.column_config.NumberColumn("value at risk £", format="£%.0f"),
                                "monetary_total": st.column_config.NumberColumn("lifetime £", format="£%.0f")})
    st.download_button("Download full list (CSV)", lst[cols].to_csv(index=False).encode(),
                       file_name="campaign_target_list.csv", mime="text/csv", disabled=lst.empty)
    st.caption("Showing the top 500 rows; the download contains all customers in the list.")

# ------------------------------------------------------------------ method
with tab_model:
    sel, t_churn, t_clv = M["selection"], M["churn"]["test"], M["clv"]["test"]
    sig = M["significance"]
    st.markdown(f"#### Honest accuracy - held-out test snapshot {M['protocol']['test_snapshot']} "
                f"(predict the next {M['protocol']['horizon_days']} days)")
    a, b = st.columns(2)
    with a:
        st.markdown("**Churn risk** (no order in next 180 days; base rate "
                    f"{M['data']['test_churn_rate'] * 100:.0f}%)")
        names = {"rule_recency_only": "Rule: days since last order", "bgnbd_expected_purchases": "BG/NBD",
                 "logistic_base": "Logistic regression (deployed)", "lightgbm_base": "LightGBM"}
        df = pd.DataFrame([{"model": n, "ROC-AUC": t_churn[k]["roc_auc"],
                            "top-decile lift": t_churn[k]["top_decile_lift"]} for k, n in names.items()])
        st.dataframe(df, hide_index=True, width="stretch", column_config={
            "ROC-AUC": st.column_config.NumberColumn(format="%.3f"),
            "top-decile lift": st.column_config.NumberColumn(format="%.2fx")})
        d = sig["churn_selected_vs_rule_auc"]
        st.caption(f"Deployed model vs the recency rule: +{d['diff']:.3f} AUC "
                   f"(95% CI {d['ci95'][0]:+.3f} to {d['ci95'][1]:+.3f}). Vs BG/NBD: no significant difference.")
    with b:
        st.markdown("**Customer lifetime value** (revenue in next 180 days)")
        names = {"naive_last_180d_spend": "Naive: last 6-month spend", "bgnbd_gamma_gamma": "BG/NBD + Gamma-Gamma",
                 "lightgbm_base": "LightGBM", "blend_bgnbd_lightgbm_plus": "Blend (deployed)"}
        df = pd.DataFrame([{"model": n, "Spearman": t_clv[k]["spearman"], "MAE £": t_clv[k]["mae"],
                            "top-10% capture %": t_clv[k]["top_decile_revenue_capture_pct"],
                            "predicted total £M": t_clv[k]["total_predicted"] / 1e6}
                           for k, n in names.items()])
        st.dataframe(df, hide_index=True, width="stretch", column_config={
            "Spearman": st.column_config.NumberColumn(format="%.3f"),
            "MAE £": st.column_config.NumberColumn(format="£%.0f"),
            "top-10% capture %": st.column_config.NumberColumn(format="%.1f%%"),
            "predicted total £M": st.column_config.NumberColumn(format="£%.2fM")})
        st.caption(f"Actual test revenue: {gbp(M['data']['test_actual_revenue'])}. The deployed blend was "
                   "chosen after seeing the test snapshot - see the README for the full story.")
    st.markdown("#### How it works")
    st.markdown(
        "* **Data:** 1.07M raw invoice lines → cleaned (duplicates, cancellations, fees, missing IDs) → "
        f"{D['cleaning']['customers']:,} customers, {D['cleaning']['invoices']:,} orders.\n"
        "* **Segments:** K-Means (k=5) on log-scaled recency / frequency / spend, named with explicit rules.\n"
        "* **Predictions:** every customer-month is a training row (features from the past, labels from "
        "the next 180 days). Validation and test snapshots are later in time than all training data.\n"
        "* **Limitations:** two years of data (one seasonal cycle), one UK retailer, 13% of revenue has no "
        "customer ID, and differences between the CLV models are small.")

st.caption("Data: UCI Machine Learning Repository, Online Retail II (CC BY 4.0). Historical data; "
           "not a live system.")
