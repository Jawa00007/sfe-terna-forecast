# PRD — Terna Imbalance Sign & Price Forecasting System

**Working name:** Sbilanciamento Forecast Engine (SFE)
**Version:** 0.1 (draft for review)
**Date:** 30 August 2026
**Status:** Discovery / pre-build

---

## 1. Summary

A forecasting service that predicts, for each settlement period of the Italian electricity market, (a) the **sign of the aggregate imbalance** published by Terna and (b) the **imbalance price** (`unbalance_price_EURxMWh`), and converts those forecasts into a **recommended procurement deviation** for a Balance Responsible Party (BRP) portfolio.

The commercial logic: a BRP's total cost for a settlement period is

```
Cost = L · P_mkt − Δ · (P_imb − P_mkt)
```

where `L` is actual portfolio load, `Δ = Q − L` is the deliberate deviation between procured volume `Q` and load, `P_mkt` is the day-ahead/intraday price and `P_imb` is the imbalance price. The incremental margin from deviating is therefore `Δ · S`, where **`S = P_imb − P_mkt`** is the signed spread.

- Area **long** (sign `+`) → `S < 0`, imbalance is cheaper → under-procure (`Δ < 0`), settle the shortfall against Terna.
- Area **short** (sign `−`) → `S > 0`, imbalance is dearer → over-procure (`Δ > 0`), sell the surplus into imbalance.

**The single most important design consequence:** the sign is a *proxy*, not the objective. Expected P&L is `E[Δ · S]`. A model with 70% sign accuracy that is wrong on the large-spread periods loses money. The system must therefore forecast the **distribution of `S`**, not just its sign, and size `Δ` proportionally to expected value and confidence.

---

## 2. Business context and the regulatory constraint

This section must be resolved before engineering starts, because it determines whether the product is viable in its stated form.

ARERA introduced a **macrozonal non-arbitrage charge** (*corrispettivo di non arbitraggio macrozonale*) explicitly to neutralise the economic advantage a dispatching user can obtain from the gap between market prices and imbalance prices, and extended it to enabled units. Further charges apply for failure to respect dispatching orders, calculated at portfolio level per area. Terna publishes these values through the `macrozonal-no-arbitrage-prices` endpoint.

Consequently:

- **R-1 (blocking):** The economic model must forecast margin **net of** the non-arbitrage charge and any applicable penalty components, not the gross spread `S`. A gross-spread backtest will systematically overstate profitability.
- **R-2 (blocking):** Obtain a written legal/regulatory opinion on the extent to which deliberate, systematic deviation from the best forecast of physical load is permissible for the BRP entity in question. Under the TIDE framework the BRP is defined as the party responsible for imbalances between the base schedule and actual injection/withdrawal; the intended behaviour is a good-faith schedule. Deliberate deviation sits in a grey zone and is at minimum charge-exposed, potentially compliance-exposed.
- **R-3:** Position the product internally as **imbalance cost optimisation and risk management** (reducing the cost of unavoidable forecast error, choosing between MI sessions and imbalance settlement) rather than as an arbitrage engine. This framing survives regulatory scrutiny and is where most of the durable value sits anyway.

Also note the reflexivity problem: at meaningful volume, the portfolio's own `Δ` contributes to the aggregate imbalance it is trying to predict. Capacity limits must be part of the design (see §9).

---

## 3. Regulatory / market-design assumptions to verify

The Italian dispatching framework is mid-reform (TIDE). These items materially affect the target variable and must be confirmed against current Terna Grid Code and ARERA resolutions before modelling:

| # | Item | Current understanding | Action |
|---|---|---|---|
| A-1 | Settlement period | 15-minute MTU under TIDE; the Terna Fees APIs accept `dataType` of `Orario` or `Quarto Orario` | Confirm which is authoritative for settlement today |
| A-2 | Geographic scope of the imbalance price | ARERA redefined it to **market zone** level per EU rules, but the `daily-prices` and `daily-macrozonal-imbalance` endpoints return **macrozone** (`NORD`, `SUD`) | **Resolve first.** Determines whether the model predicts 2 series or 6–7 |
| A-3 | Pricing rule | Single-price for all units since April 2022; price is a weighted average of activated balancing energy | Confirm exact formula and the treatment of periods with no activations ("avoided activations" value) |
| A-4 | Zone configuration | NORD, CNOR, CSUD, SUD, SICI, SARD (CALA status to confirm) | Confirm current zone list and any changes in the historical window |
| A-5 | TIDE phase | Consolidation phase from 1 Feb 2026; full regime from end-2026 | Confirm what changes on the transition and plan for a structural break |

**A-2 is the highest-priority open question and blocks the data model.** The user requirement is phrased as "zonal", but Terna's published *corrispettivi* series is macrozonal.

---

## 4. Goals and non-goals

### Goals
- G-1: Predict the sign of the aggregate imbalance per area per MTU, with calibrated probabilities.
- G-2: Predict the imbalance price per area per MTU as a full predictive distribution (quantiles), not a point estimate.
- G-3: Predict the spread `S` vs the relevant reference market price directly.
- G-4: Convert (G-1..G-3) into a recommended, risk-limited `Δ` per MTU, net of regulatory charges.
- G-5: Deliver a backtest harness that reports **economic** performance, not only statistical accuracy.

### Non-goals (v1)
- Automated order execution on MGP/MI. v1 is decision support; a human approves.
- Forecasting the client portfolio's own physical load (assumed to exist; consumed as an input).
- Balancing-market (MSD/MB) bidding optimisation for enabled units.
- Cross-border or non-Italian markets.

---

## 5. Users

| User | Need | Primary surface |
|---|---|---|
| Energy trader / scheduler | A recommended `Δ` per MTU before MGP gate closure, refreshed before each MI session | Web dashboard + morning email |
| Risk manager | Exposure, limit utilisation, worst-case scenarios, realised vs predicted P&L | Dashboard + daily report |
| Quant / data scientist | Model versions, feature importances, backtests, retraining | Notebook access + MLflow-style registry |
| Portfolio manager | Monthly attribution: how much cost was saved vs a naive perfect-hedge baseline | Monthly report |

---

## 6. Data sources

### 6.1 Terna Developer Portal (`api.terna.it`)

OAuth 2.0, client credentials. Public APIs need registration on `developer.terna.it`; **private APIs (Settlement, Misure, Modulazione Straordinaria, Anagrafiche) require a separate request through the MyTerna portal** — factor this lead time into the plan. There is no demo/sandbox account, so onboarding is on the critical path.

**Target variables (Fees / `corrispettivi`):**

| Endpoint | Provides | Role |
|---|---|---|
| `fees/v1.0/daily-macrozonal-imbalance` | `zonal_aggregate_sign`, `zonal_aggregate_unbalance_MWh`, `exchanges_MWh`, `foreign_MWh`, per macrozone | **Sign label (definitive)** |
| `fees/v1.0/daily-prices` | `base_price_EURxMWh`, `incentive_component_EURxMWh`, `unbalance_price_EURxMWh` | **Price label (definitive)** |
| `fees/v1.0/preliminary-macrozonal-imbalance` | Preliminary sign | Early signal + revision tracking |
| `fees/v1.0/preliminary-prices` | Preliminary price | Early signal + revision tracking |
| `fees/v1.0/macrozonal-no-arbitrage-prices` | Non-arbitrage charge | **Required for net-margin calculation** |
| `fees/v1.0/commercial-balance-prices` | Commercial imbalance prices | Feature |
| `fees/v1.0/zonal-exchanges` | Inter-zonal exchange values | Feature |
| `fees/v1.0/secondary-adjustment-levels` | Secondary reserve levels | Feature |

All Fees endpoints take `dateFrom` / `dateTo` in `dd/mm/yyyy` and optional `dataType` — {`Orario`, `Quarto Orario`}. Timestamps are `Europe/Rome`. Numeric fields are returned as **strings** — parse defensively.

**Feature endpoints:**

- *Market:* `load-forecast`, `forecast-transit-limits`, `transit-limits`, `transit-margins`, `secondary-reserve-requirement`, `replacement-reserve-requirement`, `rotating-reserve-requirement`, `total-reserve-requirement`, `accepted-offers`, `submitted-offers`, `prices`, `quantity`, `costs`
- *Generation:* `wind-production-forecast`, `actual-generation`, `renewable-generation`, `installed-capacity`
- *Load:* `total-load`, `market-load`
- *Transmission:* `scheduled-internal-exchange`, `physical-internal-flow`, `scheduled-foreign-exchange`, `physical-foreign-flow`
- *Outages:* `unavailability-productive-units`
- *Adequacy:* `expected-available-capacity`, `aggregate-effective-available-capacity`

Note: the Adequacy and Market endpoint descriptions on the developer portal are placeholders ("plain description to insert here"). Schemas must be discovered from the Swagger definitions and the `io-docs` test console.

### 6.2 External

- **GME** — MGP and MI zonal prices and volumes (the `P_mkt` reference leg). Confirm the reference price definition post-PUN.
- **ENTSO-E Transparency Platform** — cross-check, cross-border flows, neighbouring-country prices.
- **Weather** — ECMWF/ICON-EU or a commercial vendor. Zone-weighted temperature, irradiance, wind speed at hub height, plus **forecast-error proxies** (spread between successive NWP runs), which are among the strongest predictors of system imbalance.
- **Calendar** — Italian national and regional holidays, school calendars, DST transitions.

### 6.3 Data quality requirements

- **DQ-1:** Store **preliminary and definitive values separately with publication timestamps.** Terna has publicly corrected imbalance prices after publication following input-data anomalies. Never overwrite; append with vintage.
- **DQ-2:** All training and backtesting must use a **point-in-time (as-of) view** — only data whose `publication_date` precedes the decision timestamp. This is the single most common source of illusory backtest performance in this domain.
- **DQ-3:** Handle DST (23- and 25-hour days; 92/100 quarter-hour days) explicitly. Store UTC internally, present Europe/Rome.
- **DQ-4:** Detect and flag structural breaks (zone reconfiguration, TIDE phase transitions, MTU change from hourly to quarter-hourly) as regime markers available to the model.

---

## 7. Functional requirements

**Ingestion**
- F-1: Scheduled pull of all listed Terna endpoints with OAuth token refresh, retry with backoff, and per-endpoint watermarking.
- F-2: Backfill of ≥ 4 years of history where available, with the regime markers from DQ-4.
- F-3: Idempotent, append-only landing zone; raw payloads retained.

**Feature store**
- F-4: Point-in-time correct feature store keyed on (area, MTU start, as-of timestamp).
- F-5: Feature families: calendar; lagged imbalance sign/volume/price with realistic publication lags; residual load (load forecast minus wind and solar forecast); reserve requirements and margins; transit limits and saturation; unit outages; NWP forecast-revision volatility; MGP/MI price and price shape; neighbouring-zone state.

**Models**
- F-6: **Sign model** — binary classifier per area per MTU producing a calibrated probability. Baseline: gradient-boosted trees (LightGBM/XGBoost). Calibration via isotonic or Platt scaling, validated with reliability diagrams.
- F-7: **Spread model** — quantile regression on `S = P_imb − P_mkt` producing at minimum the 10/25/50/75/90th percentiles. Baseline: LightGBM with pinball loss; candidate upgrade: distributional GBM or a small temporal model (TFT / N-HiTS) once the tabular baseline is beaten.
- F-8: **Price model** — derived as `P_mkt_forecast + S_forecast` rather than modelled independently, so the two legs stay coherent. Absolute imbalance price is reported for interpretability.
- F-9: Multi-horizon: a forecast for each decision point — pre-MGP-gate-closure (D-1), and refreshed ahead of each MI session up to the last intraday gate closure. Horizons are separate models or a horizon feature; do not train one model and apply it at all horizons.
- F-10: Retraining on a rolling window with automatic drift detection; a manual override to force retrain after a regulatory change.

**Decision layer**
- F-11: Convert the spread distribution into a recommended `Δ` subject to: expected net margin threshold, portfolio volume limits, per-MTU and daily VaR/CVaR limits, and a hard cap as a percentage of load.
- F-12: Explicitly subtract forecast non-arbitrage charges and expected penalty components before recommending a position.
- F-13: Refuse to recommend a position where the predictive distribution is wide relative to the expected margin (abstention is a valid and frequently correct output).

**Interface**
- F-14: Dashboard: per-area, per-MTU sign probability, spread fan chart, recommended `Δ`, limit utilisation, and the drivers behind each recommendation.
- F-15: REST API for downstream systems.
- F-16: Daily P&L attribution: realised vs predicted, decomposed into sign error, magnitude error, and charge effects.
- F-17: Full audit log of every recommendation with the input feature vector and model version (needed for both debugging and any regulatory query).

---

## 8. Metrics

**Statistical (necessary, not sufficient)**
- Sign: balanced accuracy, AUC, **Brier score**, calibration error. Baseline to beat: persistence (same MTU, previous day) and the seasonal-climatological base rate.
- Price/spread: MAE, RMSE, **pinball loss** across quantiles, PICP (coverage of the 80% interval).

**Economic (the ones that decide go/no-go)**
- **Primary: net margin in €/MWh** of portfolio volume vs a perfect-hedge (`Δ = 0`) baseline, after all charges.
- Value-weighted sign hit rate (weight each period by `|S|`).
- Realised Sharpe of the daily P&L series.
- Maximum drawdown; worst single MTU; worst single day.
- Fraction of periods where the system abstains.

**Go/no-go gate:** the strategy proceeds past the pilot only if net margin after charges is positive and statistically significant on a **fully out-of-time** test period, with no point-in-time leakage, over at least 6 months spanning both summer and winter.

---

## 9. Backtesting

- B-1: Walk-forward, expanding-window evaluation. No random shuffling, no k-fold on time series.
- B-2: The final test period is held out entirely and touched once.
- B-3: Every backtest replays the **as-of** data view at each historical decision timestamp (F-4).
- B-4: Market-impact model: assume the portfolio's own `Δ` shifts the aggregate imbalance and therefore the spread. Estimate elasticity empirically and include it; without this, the backtest overstates capacity.
- B-5: Sensitivity of results to the non-arbitrage charge assumption, reported as a range.
- B-6: Ablation vs the naive baselines from §8 — the reference is "would a scheduler with a rule of thumb have done as well?"

---

## 10. Architecture (proposed)

```
Terna API ─┐
GME        ─┼─ Ingestion (Airflow/Prefect) ─┼ Raw store (S3/Parquet, append-only, vintaged)
ENTSO-E    ─┤                                        │
Weather   ─┘                                        ▼
                                          Feature store (point-in-time)
                                                    │
                              ┌──────────────────────┼──────────────────────┐
                              ▼                     ▼                     ▼
                       Sign classifier      Spread quantile reg.    P_mkt forecast
                              └──────────────────────┼──────────────────────┘
                                                    ▼
                                        Decision layer (Δ, limits, charges)
                                                    │
                                    ┌───────────────┼───────────────┐
                                    ▼               ▼               ▼
                              Dashboard        REST API        Audit log
```

**Non-functional requirements**
- NF-1: Forecasts for day D available no later than 09:00 D-1 (comfortably before MGP gate closure), and within 15 minutes of each intraday refresh trigger.
- NF-2: 99.5% availability during the decision windows.
- NF-3: Graceful degradation — if an upstream feed is stale, fall back to a reduced-feature model and flag the recommendation as degraded rather than failing silently.
- NF-4: All model versions reproducible from a commit hash plus a data snapshot ID.

---

## 11. Rollout

| Phase | Duration | Content | Exit criterion |
|---|---|---|---|
| 0. Access & feasibility | 3–4 wks | Terna developer registration; MyTerna request for private APIs; resolve A-1..A-5; obtain regulatory opinion (R-2) | Data flowing; target variable unambiguously defined; legal sign-off |
| 1. Data foundation | 4–6 wks | Ingestion, backfill, point-in-time feature store, EDA on the spread distribution | Reproducible historical dataset with vintages |
| 2. Baseline models | 4 wks | Sign classifier + spread quantile regression; beat persistence | Statistically significant improvement over baselines |
| 3. Economic backtest | 3 wks | Decision layer, charges, market impact, walk-forward | Net-of-charges margin positive on out-of-time data |
| 4. Shadow mode | 8–12 wks | Recommendations generated and logged but not acted on | Live performance tracks backtest within tolerance |
| 5. Limited pilot | 12 wks | Small capped volume, human approval on every position | Realised P&L positive; no regulatory issues raised |
| 6. Scale | — | Raise limits incrementally | — |

---

## 12. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Non-arbitrage charge eliminates the margin | **High** | Model net margin from day one (R-1); kill the strategy leg early if it fails, keep the cost-optimisation leg |
| Deliberate deviation is regulatorily impermissible | **High** | R-2 legal opinion in Phase 0, before build |
| TIDE regime change invalidates trained models | High | Regime markers; short retraining windows; expect a rebuild around the end-2026 transition |
| Backtest leakage produces phantom profits | High | DQ-2 and B-3 are non-negotiable; independent review of the backtest harness |
| Spread distribution is heavy-tailed; a few periods dominate P&L | Medium | Quantile models, CVaR limits, per-MTU caps |
| No sandbox — API behaviour discovered in production | Medium | Defensive parsing, contract tests against recorded payloads, staged rollout |
| Published values are revised after the fact | Medium | DQ-1 vintaging; measure preliminary-to-definitive revision distribution and treat it as a risk term |
| Strategy is self-defeating at scale | Medium | Market-impact model (B-4); explicit capacity ceiling |

---

## 13. Open questions

1. **A-2: macrozone or market zone?** Which series is the actual settlement basis for this portfolio today?
2. Which entity is the BRP, and what is its regulatory standing for deliberate deviation?
3. Hourly or quarter-hourly MTU for the portfolio's current settlement?
4. What is the portfolio volume, and what deviation caps are acceptable to risk?
5. Which reference `P_mkt` — MGP zonal, volume-weighted MGP+MI, or realised procurement price?
6. Does an internal load forecast already exist, and what is its accuracy? (It bounds everything downstream.)
7. Are the private Settlement/Misure APIs needed, or is the public Fees catalogue sufficient?
8. Budget and appetite for commercial NWP data vs open ECMWF products.

---

## 14. Sources

- Terna Developer Portal — API catalogue and Fees endpoint specifications: `https://developer.terna.it/docs/read/apis_catalog`
- Terna Data Portal — *Corrispettivi*: `https://dati.terna.it/corrispettivi`
- ARERA — imbalance reform, single price, market-zone scope, macrozonal non-arbitrage charge
- ARERA — TIDE (Testo Integrato Dispacciamento Elettrico), phased implementation 2025–2027
