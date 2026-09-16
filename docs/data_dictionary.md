# Data dictionary

## Landing zone (long format)

Every landed cell — one row per `(period, area, field, vintage)` — carries the metadata
columns defined in `sfe.io.storage.META_COLUMNS`:

| Column | Type | Meaning |
|---|---|---|
| `_source` | str | `terna` \| `gme` \| `entsoe` \| `weather` \| `calendar` |
| `_endpoint` | str | Endpoint / feed name, e.g. `daily-prices` |
| `_area` | str | Macrozone or market-zone code |
| `_area_level` | str | `macrozone` \| `zone` (from `conf/zones.yaml:target_level`) |
| `_mtu_start` | timestamp, UTC | Start of the settlement period; `NULL` if unparseable |
| `_resolution` | str | `PT1H` \| `PT15M` \| `P1D` |
| `_vintage` | str | `preliminary` \| `definitive` |
| `_publication_ts` | timestamp, UTC | When the source published this value (drives point-in-time) |
| `_ingested_at` | timestamp, UTC | When we pulled it |
| `_payload_hash` | str (sha1) | Idempotency key over the stable source record + field |
| `_raw` | str (JSON) | Verbatim source record (raw-payload retention, F-3) |
| `field` | str | Source value field name, e.g. `unbalance_price_EURxMWh` |
| `value_raw` | str | Value exactly as received (may be a decimal-comma string) |
| `value_num` | float | Defensively parsed numeric, or `NULL` |

## Target endpoints (PRD §6.1)

| Endpoint | Role | Key fields | Value fields |
|---|---|---|---|
| `daily-macrozonal-imbalance` | definitive sign | Date, Macrozone | `zonal_aggregate_sign`, `zonal_aggregate_unbalance_MWh`, `exchanges_MWh`, `foreign_MWh` |
| `daily-prices` | definitive price | Date, Macrozone | `base_price_EURxMWh`, `incentive_component_EURxMWh`, `unbalance_price_EURxMWh` |
| `preliminary-macrozonal-imbalance` / `preliminary-prices` | early signal + revision tracking | | |
| `macrozonal-no-arbitrage-prices` | charge (net-margin) | Date, Macrozone | non-arbitrage price |

> Field lists are from the PRD and must be reconciled against the live Swagger + io-docs in
> Phase 0. Keep the defensive parsing regardless.

## Derived quantities

- `S = unbalance_price_EURxMWh − P_mkt` — signed spread (reference `P_mkt` per open question #5).
- Sign convention: `+1` area long (imbalance cheaper, `S < 0`), `−1` area short (`S > 0`).
