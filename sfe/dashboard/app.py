"""Operational monitoring dashboard (PRD F-14 + Phase 4 shadow-mode observability).

Five tabs:
  Forecast          per-MTU sign probability, spread fan, recommended Delta, drivers (F-14)
  Data & Feeds      landing dataset row counts + publication staleness (NF-3)
  Models            per-horizon registration status, age, and an on-demand drift check (F-10)
  Backtest Runs     the CLI run log (train_baseline / run_backtest / retrain_check)
  Audit & P&L       browse the audit log (F-17) and realised-vs-predicted attribution (F-16)

    .venv/Scripts/streamlit run sfe/dashboard/app.py
"""

from __future__ import annotations

import os
from datetime import date, timedelta

import pandas as pd
import streamlit as st

from sfe.curate.timebase import utc_to_rome
from sfe.features.spec import HORIZON_OFFSET_H
from sfe.ingest.watermark import all_watermarks
from sfe.io.storage import get_storage
from sfe.models.registry import load_model_meta, model_exists, read_runs
from sfe.models.retrain import model_age_days, should_retrain
from sfe.monitoring.audit import AuditLog
from sfe.monitoring.drift import staleness_report
from sfe.serve.inference import generate_bundle

st.set_page_config(page_title="SFE - Monitoring", layout="wide")


def _check_password() -> bool:
    """Optional gate: set DASHBOARD_PASSWORD to require it before rendering anything."""
    required = os.environ.get("DASHBOARD_PASSWORD")
    if not required:
        return True
    if st.session_state.get("authed"):
        return True
    entered = st.text_input("Password", type="password")
    if entered == required:
        st.session_state["authed"] = True
        st.rerun()
    elif entered:
        st.error("Wrong password")
    return False


if not _check_password():
    st.stop()

st.title("Sbilanciamento Forecast Engine - monitoring")
st.caption(
    "Shadow-mode dashboard. Phase 0-4 build; no live Terna data yet - reads whatever has "
    "been landed (mock or real) under the configured storage root."
)

_FEEDS = [
    "landing/terna/fees/daily-prices",
    "landing/terna/fees/daily-macrozonal-imbalance",
    "landing/terna/fees/preliminary-prices",
    "landing/terna/fees/preliminary-macrozonal-imbalance",
    "landing/terna/fees/macrozonal-no-arbitrage-prices",
    "landing/terna/market/load-forecast",
    "landing/terna/generation/wind-production-forecast",
    "landing/gme/mgp-zonal-prices",
]

tab_forecast, tab_data, tab_models, tab_backtest, tab_audit = st.tabs(
    ["Forecast", "Data & Feeds", "Models", "Backtest Runs", "Audit & P&L"]
)


# ---------------------------------------------------------------- Forecast --
def _render_forecast_tab() -> None:
    c1, c2, c3 = st.columns([2, 1, 1])
    dd = c1.date_input("Delivery day", value=date.today() + timedelta(days=1), key="fc_day")
    horizon = c2.selectbox("Horizon", list(HORIZON_OFFSET_H), index=0, key="fc_horizon")
    run = c3.button("Generate forecast", type="primary", key="fc_run")

    if not run:
        st.info("Pick a delivery day and horizon, then Generate forecast.")
        return

    with st.spinner("Assembling features and forecasting..."):
        bundle = generate_bundle(dd, horizon)
    meta, table, drivers = bundle.to_dict(), bundle.table, bundle.drivers

    if meta["degraded"]:
        st.warning("DEGRADED: " + "; ".join(meta["degraded_reasons"]))
    st.caption(f"as of {meta['as_of']}  -  model {meta['model_version']}")

    if table.empty:
        st.error("No forecast rows.")
        return

    table = table.copy()
    table["hour"] = table["_mtu_start"].map(
        lambda t: utc_to_rome(pd.Timestamp(t).to_pydatetime()).hour
    )
    areas = sorted(table["_area"].unique())
    area = st.selectbox("Area", areas, key="fc_area")
    a = table[table["_area"] == area].sort_values("_mtu_start")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Active positions", int((~a["abstain"]).sum()))
    c2.metric("Sum |Delta| (MWh)", round(a["delta_mwh"].abs().sum(), 1))
    c3.metric("Exp. net margin (EUR)", round(a["expected_net_margin_eur"].sum(), 0))
    c4.metric("Max load-cap util", f"{a['util_pct_load_cap'].max():.0%}")

    st.subheader("Spread fan  (S = P_imb - P_mkt)")
    st.line_chart(a.set_index("hour")[["S_p10", "S_p25", "S_p50", "S_p75", "S_p90"]])

    st.subheader("Sign probability  P(area long)")
    st.bar_chart(a.set_index("hour")[["sign_prob"]])

    st.subheader("Recommended Delta (MWh)  -  negative = under-procure")
    st.bar_chart(a.set_index("hour")[["delta_mwh"]])

    st.subheader("Per-MTU detail")
    st.dataframe(
        a[
            [
                "hour", "sign_prob", "S_p10", "S_p50", "S_p90", "delta_mwh", "abstain",
                "reason", "charge_eur_per_mwh", "expected_net_margin_eur",
                "cvar_eur_per_mwh", "util_pct_load_cap", "limit_binding",
            ]
        ],
        width="stretch",
        hide_index=True,
    )

    st.subheader("Drivers  (sign-model feature importance)")
    st.dataframe(drivers, hide_index=True, width="stretch")

    if st.button("Record this bundle to the audit log", key="fc_audit"):
        ids = AuditLog().record(bundle)
        st.success(f"recorded {len(ids)} rows")


# -------------------------------------------------------------- Data & Feeds --
@st.cache_data(ttl=60, show_spinner=False)
def _dataset_summary(datasets: tuple[str, ...]) -> pd.DataFrame:
    storage = get_storage()
    rows = []
    for ds in datasets:
        if not storage.exists(ds):
            rows.append({"dataset": ds, "rows": 0, "first_mtu": None, "last_mtu": None})
            continue
        df = storage.read_dataset(ds, columns=["_mtu_start"])
        mtu = pd.to_datetime(df["_mtu_start"], utc=True)
        rows.append(
            {
                "dataset": ds,
                "rows": len(df),
                "first_mtu": mtu.min(),
                "last_mtu": mtu.max(),
            }
        )
    return pd.DataFrame(rows)


def _render_data_tab() -> None:
    st.subheader("Feed publication staleness (NF-3)")
    max_age = st.slider("Stale threshold (hours)", 6, 96, 30, key="data_stale_h")
    stale = staleness_report(_FEEDS, max_age_hours=max_age)
    n_stale = int(stale["stale"].sum())
    if n_stale:
        st.warning(f"{n_stale} feed(s) stale or missing")
    else:
        st.success("All feeds within threshold")
    st.dataframe(stale, width="stretch", hide_index=True)

    st.subheader("Landing dataset row counts")
    summary = _dataset_summary(tuple(_FEEDS))
    st.dataframe(summary, width="stretch", hide_index=True)

    st.subheader("Ingestion watermarks")
    wm = all_watermarks()
    if wm:
        st.dataframe(
            pd.DataFrame(sorted(wm.items()), columns=["endpoint", "watermark"]),
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("No watermarks recorded yet - run sfe.scripts.backfill.")

    if st.button("Refresh", key="data_refresh"):
        _dataset_summary.clear()
        st.rerun()


# ------------------------------------------------------------------ Models --
def _render_models_tab() -> None:
    st.subheader("Registered models per horizon (F-9)")
    rows = []
    for h in HORIZON_OFFSET_H:
        row = {"horizon": h}
        for kind in ("sign", "spread"):
            name = f"{kind}_{h}"
            row[f"{kind}_registered"] = model_exists(name)
            meta = load_model_meta(name) or {}
            row[f"{kind}_age_days"] = model_age_days(h, kind=kind)
            row[f"{kind}_n_features"] = meta.get("n_features")
            row[f"{kind}_train_window"] = (
                f"{meta.get('train_start', '?')} .. {meta.get('train_end', '?')}"
                if meta
                else "-"
            )
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.divider()
    st.subheader("Drift / retrain check (F-10)")
    st.caption(
        "Rebuilds a reference and a current feature window and compares them (PSI). "
        "Can take a while on a large landing store - run on demand, not on every page load."
    )
    c1, c2, c3, c4 = st.columns(4)
    horizon = c1.selectbox("Horizon", list(HORIZON_OFFSET_H), key="mdl_horizon")
    end = c2.date_input("As of (end of current window)", value=date.today(), key="mdl_end")
    max_age = c3.number_input("Max age (days)", value=14.0, key="mdl_max_age")
    drift_frac = c4.number_input("Drift alert fraction", value=0.2, step=0.05, key="mdl_drift_frac")

    if st.button("Check now", type="primary", key="mdl_check"):
        with st.spinner("Rebuilding reference + current feature windows..."):
            decision = should_retrain(
                horizon, end=end, max_age_days=max_age, drift_alert_fraction=drift_frac
            )
        if decision.retrain:
            st.warning("RETRAIN recommended: " + "; ".join(decision.reasons))
        else:
            st.success("No retrain triggered")
        age_str = (
            f"{decision.model_age_days:.1f}" if decision.model_age_days is not None else "n/a"
        )
        st.metric("Model age (days)", age_str)
        if not decision.drift.empty:
            st.dataframe(decision.drift, width="stretch", hide_index=True)
        st.caption(
            "To retrain from the CLI: "
            f"`python -m sfe.scripts.retrain_check --horizons {horizon} --apply`"
        )


# -------------------------------------------------------------- Backtest Runs --
def _render_backtest_tab() -> None:
    st.caption(
        "Populated by `sfe.scripts.train_baseline`, `run_backtest` and `retrain_check` "
        "(each calls `sfe.models.registry.log_run`). Nothing is run from here."
    )
    runs = read_runs()
    if runs.empty:
        st.info(
            "No runs logged yet. Try:\n\n"
            "```\npython -m sfe.scripts.run_backtest --from 2024-09-20 --to 2025-01-20 "
            "--min-train-days 60 --test-days 20 --fast\n```"
        )
        return

    kinds = sorted(runs["name"].unique())
    picked = st.multiselect("Run type", kinds, default=kinds, key="bt_kinds")
    view = runs[runs["name"].isin(picked)]
    st.dataframe(view, width="stretch", hide_index=True)

    margin_col = "metric.net_margin_eur_per_mwh"
    if margin_col in view.columns and view[margin_col].notna().any():
        st.subheader("Net margin EUR/MWh over successive runs")
        chart = view[["logged_at", margin_col]].dropna().sort_values("logged_at")
        st.line_chart(chart.set_index("logged_at"))


# ------------------------------------------------------------------ Audit --
@st.cache_data(ttl=30, show_spinner=False)
def _read_audit() -> pd.DataFrame:
    return AuditLog().read()


def _render_audit_tab() -> None:
    log = _read_audit()
    if log.empty:
        st.info(
            "Audit log is empty. Generate a forecast in the Forecast tab and click "
            "'Record this bundle to the audit log', or run `sfe.scripts.shadow_run`."
        )
        return

    log = log.copy()
    log["delivery_day"] = pd.to_datetime(log["delivery_day"]).dt.date
    days = sorted(log["delivery_day"].unique())
    c1, c2, c3 = st.columns(3)
    day_from = c1.selectbox("From", days, index=0, key="au_from")
    day_to = c2.selectbox("To", days, index=len(days) - 1, key="au_to")
    only_active = c3.checkbox("Active positions only", value=False, key="au_active")

    view = log[(log["delivery_day"] >= day_from) & (log["delivery_day"] <= day_to)]
    if only_active:
        view = view[~view["abstain"]]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", len(view))
    c2.metric("Abstention", f"{view['abstain'].mean():.0%}" if len(view) else "n/a")
    c3.metric("Degraded rows", int(view["degraded"].sum()))
    c4.metric("Sum expected net margin (EUR)", round(view["expected_net_margin_eur"].sum(), 0))

    st.subheader("Recorded recommendations")
    st.dataframe(
        view.drop(columns=["features_json"]).sort_values("_mtu_start", ascending=False).head(500),
        width="stretch",
        hide_index=True,
    )
    picked = st.selectbox(
        "Inspect one row's feature vector", ["-"] + view["audit_id"].tolist(), key="au_pick"
    )
    if picked != "-":
        row = view[view["audit_id"] == picked].iloc[0]
        st.json(row["features_json"] or "{}")

    st.divider()
    st.subheader("Realised vs predicted P&L attribution (F-16)")
    st.caption(
        "Only scores rows whose delivery day's outcome has since been landed "
        "(`spread_S` known); rows still in the future show as unscored."
    )
    if st.button("Compute attribution for this range", key="au_attr"):
        from sfe.backtest.walkforward import build_actuals
        from sfe.monitoring.pnl_attribution import attribute, daily_attribution

        keys = view[["_area", "_mtu_start"]].drop_duplicates()
        with st.spinner("Joining against realised outcomes..."):
            actual = build_actuals(keys)
            att = attribute(view, actual)
            daily = daily_attribution(att)
        if daily.empty:
            st.info("No delivery day in this range has a realised outcome yet.")
        else:
            st.dataframe(daily, width="stretch", hide_index=True)
            st.subheader("Realised vs expected P&L by day")
            st.bar_chart(daily.set_index("delivery_day")[["pnl_realized_eur", "pnl_expected_eur"]])
            st.subheader("Error decomposition (spread vs charge)")
            st.bar_chart(daily.set_index("delivery_day")[["err_spread_eur", "err_charge_eur"]])


with tab_forecast:
    _render_forecast_tab()
with tab_data:
    _render_data_tab()
with tab_models:
    _render_models_tab()
with tab_backtest:
    _render_backtest_tab()
with tab_audit:
    _render_audit_tab()
