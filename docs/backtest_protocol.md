# Backtest protocol (PRD §9, §8)

Not yet implemented — this records the rules the Phase 3 harness must enforce so they are
agreed before code is written.

## Non-negotiable

- **B-1** Walk-forward, expanding window. No random shuffling, no k-fold on time.
- **B-2** The final out-of-time test period is held out entirely and touched **once**.
  ≥ 6 months, spanning both summer and winter.
- **B-3** Every historical decision timestamp replays the **as-of** data view
  (`sfe.curate.vintage.as_of_view`): only values with `_publication_ts < decision_ts`.
  `assert_no_leakage` runs on every assembled feature frame.
- **B-4** Market-impact model: the portfolio's own `Δ` shifts the aggregate imbalance and
  therefore `S`. Elasticity estimated empirically; without it, capacity is overstated.
- **B-5** Report sensitivity to the non-arbitrage charge as a range
  (`conf/charges.yaml:sensitivity_multipliers`).
- **B-6** Ablate against the naive baselines below.

## Baselines to beat

- Persistence: same MTU, previous day.
- Seasonal climatology: base rate by (month, hour, area).

## Metrics

Statistical (necessary, not sufficient): sign — balanced accuracy, AUC, Brier, calibration
error; spread — MAE, RMSE, pinball loss per quantile, PICP of the 80% interval.

Economic (decide go/no-go):

- **Primary — net margin €/MWh vs a perfect-hedge (`Δ = 0`) baseline, after all charges.**
- Value-weighted sign hit rate (weight each period by `|S|`).
- Realised Sharpe of the daily P&L series.
- Max drawdown; worst single MTU; worst single day.
- Fraction of periods the system abstains.

## Go / no-go gate

Proceed past the pilot only if net margin after charges is **positive and statistically
significant** on the fully out-of-time test period, with no point-in-time leakage, over
≥ 6 months spanning summer and winter. Independent review of the harness for leakage before
the result is trusted.
