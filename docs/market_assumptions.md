# Market-design assumptions (PRD §3, A-1…A-5)

Owner: _TBD_ · Status: **OPEN — blocks the target variable definition**

Every row must be confirmed against the current Terna Grid Code and ARERA resolutions
before modelling starts. Update `conf/regimes.yaml` and flip `confirmed: true` per entry as
each is signed off. `sfe.curate.regimes.unconfirmed_regimes()` lists what remains.

| # | Question | Current assumption (placeholder) | Confirmed? | Source / notes |
|---|---|---|---|---|
| A-1 | Settlement MTU authoritative today | Hourly (`Orario`) until the TIDE 15-min cutover | ☐ | |
| A-2 | Geographic scope of the imbalance price | **Macrozone** (`NORD`, `SUD`) per the published `daily-*` series; code supports switching to market zone via `conf/zones.yaml:target_level` | ☐ | **Highest priority.** Determines 2 vs 6–7 target series |
| A-3 | Single-price formula + no-activation ("avoided activations") treatment | Single price since 2022-04; weighted average of activated balancing energy | ☐ | |
| A-4 | Current zone list + historical changes in the window | `NORD, CNOR, CSUD, SUD, SICI, SARD`; CALA merged (date TBC) | ☐ | |
| A-5 | TIDE phase transitions and what changes at each | Consolidation from 2026-02-01; full regime end-2026 → structural break / model rebuild | ☐ | |

## Blocking regulatory items (PRD §2)

- **R-1** — economic model must be net of the macrozonal non-arbitrage charge + penalties.
  Tracked in `conf/charges.yaml`; source endpoint `macrozonal-no-arbitrage-prices`.
- **R-2** — written legal opinion on the permissibility of deliberate systematic deviation
  for the BRP entity. Owner: _TBD_. **Phase 5 pilot cannot start without this.**
- **R-3** — internal positioning as imbalance cost optimisation / risk management.

## Open questions (PRD §13)

1. A-2 resolution for _this portfolio_.
2. Which entity is the BRP; its regulatory standing for deviation.
3. Hourly or quarter-hourly settlement for the portfolio today.
4. Portfolio volume and acceptable deviation caps.
5. Reference `P_mkt`: MGP zonal / volume-weighted MGP+MI / realised procurement price.
6. Does an internal load forecast exist; what is its accuracy.
7. Are the private Settlement/Misure APIs needed, or is the public Fees catalogue enough.
8. Budget for commercial NWP data vs open ECMWF products.
