# Sbilanciamento Forecast Engine (SFE)

Forecasts the **sign** and **price** of the Italian aggregate imbalance published by Terna,
and converts them into a **risk-limited, net-of-charges procurement deviation `Δ`** for a
Balance Responsible Party portfolio.

- Product spec: [`PRD_Terna_Imbalance_Forecasting.md`](PRD_Terna_Imbalance_Forecasting.md)
- Build plan: `.claude/plans/store-and-understand-this-melodic-swing.md`

## Status

Phase 0–4. No Terna API access yet — everything below is developed and tested offline
against the local mock server, `sfe.testing.mock_terna`.

Working end to end on synthetic payloads:

- **Ingest & data** — OAuth2 client + resilient GET, vintaged append-only landing,
  Europe/Rome DST arithmetic, point-in-time / as-of views, Italian market calendar.
- **Features** — point-in-time feature store (calendar, lagged-target with realistic
  publication lags, residual load) + spread-distribution EDA report.
- **Models** — calibrated LightGBM sign classifier + LightGBM quantile-regression spread
  model, scored vs persistence / seasonal-climatology baselines on a chronological split.
- **Decision + backtest** — charges (R-1), market-impact elasticity (P-7), closed-form
  sizing, VaR/CVaR + volume limits, abstention; walk-forward **economic** backtest: net
  margin €/MWh after charges vs a perfect hedge, charge-sensitivity band, rule-of-thumb
  ablation, go/no-go gate.
- **Shadow mode** — inference pipeline producing per-MTU sign probability, spread and
  price fans, risk-limited Δ with limit utilisation and drivers, plus a `degraded` flag on
  stale feeds; REST API (`sfe.api.main:app`), append-only audit log with the full feature
  vector, daily P&L attribution (spread / charge / sign decomposition), and PSI drift +
  feed-staleness monitors.
- **Monitoring dashboard** (`sfe/dashboard/app.py`, Streamlit) — five tabs: *Forecast*
  (F-14), *Data & Feeds* (landing row counts + publication staleness), *Models* (per-horizon
  registration/age + an on-demand drift check), *Backtest Runs* (the CLI run log), and
  *Audit & P&L* (browse the audit log, compute realised-vs-predicted attribution for
  delivery days whose outcome has since landed).
- **Per-horizon models + retraining (F-9, F-10)** — the sign/spread models are keyed and
  registered per decision horizon (`sign_D-1`, `spread_MI1`, ...), each carrying training
  metadata (window, row/feature counts). A retrain policy triggers on model age, PSI
  feature drift between a reference and current window, or a missing model, with a manual
  force override for post-regulatory-change rebuilds.

**Real Terna data** arrives in **Phase 0** (developer-portal + MyTerna registration — a
business task, not code); the ≥4-year backfill runs in **Phase 1** once credentials land.
The **dashboard** and REST API landed here in **Phase 4 — Shadow mode**; running them for
real still waits on Phase 0/1 data. Remaining: Phase 5 limited pilot (gated on the R-2
legal opinion). See `.claude/plans/`.

## Core design constraint

The imbalance **sign is a proxy, not the objective**. Expected P&L is `E[Δ·S]` net of the
ARERA macrozonal non-arbitrage charge, where `S = P_imb − P_mkt`. The system forecasts the
**distribution of `S`** and sizes `Δ` to expected value and confidence. Everything is
measured in **€/MWh net of charges** on **fully out-of-time** data with **no point-in-time
leakage**.

## Quickstart

```bash
uv venv --python 3.11
uv pip install -e ".[dev,serve]"

# run the offline test suite
.venv/Scripts/pytest

# exercise the ingest path against the mock portal
.venv/Scripts/python -m sfe.testing.mock_terna &
SFE_TERNA__BASE_URL=http://127.0.0.1:8900 \
  SFE_TERNA__TOKEN_URL=http://127.0.0.1:8900/oauth/accessToken \
  SFE_TERNA__CLIENT_ID=mock SFE_TERNA__CLIENT_SECRET=mock \
  .venv/Scripts/python -m sfe.scripts.backfill --endpoint daily-prices \
  --from 2025-01-01 --to 2025-01-07

# dry-run the full target-endpoint backfill plan (no HTTP)
.venv/Scripts/python -m sfe.scripts.backfill --targets --from 2021-01-01 --to 2025-12-31 --dry-run

# build the point-in-time feature store for a delivery-day range, then the EDA report
.venv/Scripts/python -m sfe.scripts.build_features --from 2025-01-13 --to 2025-01-19 --horizon D-1
.venv/Scripts/python -m sfe.scripts.eda_spread

# train + evaluate the Phase 2 baselines (sign classifier + spread quantile regression)
.venv/Scripts/python -m sfe.scripts.train_baseline --from 2024-08-05 --to 2025-02-25

# Phase 3 walk-forward economic backtest: net margin EUR/MWh after charges, go/no-go gate
.venv/Scripts/python -m sfe.scripts.run_backtest --from 2024-09-20 --to 2025-01-20 \
  --min-train-days 60 --test-days 20 --fast

# Phase 4 shadow mode: daily forecast + audit log
.venv/Scripts/python -m sfe.scripts.shadow_run --date 2025-01-30 --horizon D-1

# REST API  (uvicorn) and dashboard  (pip install '.[serve,dashboard]')
.venv/Scripts/python -m uvicorn sfe.api.main:app --port 8000
.venv/Scripts/streamlit run sfe/dashboard/app.py

# train + register the model set for every horizon (F-9)
.venv/Scripts/python -m sfe.scripts.train_all_horizons --end 2025-01-25 --lookback-days 90

# drift/age-triggered retrain check (F-10); --apply actually retrains triggered horizons
.venv/Scripts/python -m sfe.scripts.retrain_check --end 2025-01-25 --apply
```

## Deploying the dashboard

`render.yaml` + `sfe/scripts/seed_demo_data.py` publish the monitoring dashboard as a
public Render web service, self-seeded with synthetic data (no real Terna connection).
See [`docs/deploy_render.md`](docs/deploy_render.md).

## Layout

| Path | Role |
|---|---|
| `conf/` | Endpoint catalogue, zones (A-2), regime markers (A-1/A-4/A-5), charge params (R-1) |
| `sfe/config.py` | Env + YAML settings |
| `sfe/io/` | Append-only vintaged Parquet store + DuckDB catalog |
| `sfe/ingest/` | OAuth2 Terna client, endpoint specs, watermarks, vintaged landing writer, flows, non-Terna providers (`external.py`) |
| `sfe/curate/` | Defensive parsing, Europe/Rome↔UTC DST arithmetic, point-in-time views, regimes, Italian calendar |
| `sfe/features/` | Point-in-time feature store: `spec` (DecisionContext), `targets`, `calendar`, `lags`, `residual_load`, `store` |
| `sfe/models/` | `dataset` (training frame + chronological split), `sign`, `spread`, `price`, `baselines`, `metrics`, `registry` (+ metadata sidecar), `retrain` (F-10 policy) |
| `sfe/decision/` | `charges` (R-1), `market_impact` (P-7), `sizing`, `limits` (VaR/CVaR, caps), `abstain`, `policy` |
| `sfe/backtest/` | `walkforward` (expanding window, as-of replay), `economic` (net-margin report + go/no-go), `sensitivity` (B-5), `ablation` (B-6) |
| `sfe/serve/` | `inference` — forecast bundle (fans, Δ, utilisation, drivers, degraded flag) + model load/train |
| `sfe/monitoring/` | `audit` (F-17), `pnl_attribution` (F-16), `drift` (PSI + staleness, NF-3) |
| `sfe/api/` | FastAPI app (F-15) · `sfe/dashboard/` Streamlit app (F-14) |
| `sfe/testing/` | Mock Terna portal + synthetic payloads |
| `sfe/scripts/` | `backfill`, `build_features`, `eda_spread`, `train_baseline`, `run_backtest`, `shadow_run`, `train_all_horizons`, `retrain_check` |
| `tests/` | `unit/` + `contract/` (parsing, feature PIT correctness, model / backtest / serve / API / retrain smoke) |

Remaining: Phase 5 limited pilot — gated on the R-2 legal opinion, not on code. See the
build plan for the rollout detail.
