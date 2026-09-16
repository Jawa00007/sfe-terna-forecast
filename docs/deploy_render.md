# Deploying the dashboard to Render

Publishes `sfe/dashboard/app.py` as a public web service using the
[`render.yaml`](../render.yaml) Blueprint at the repo root.

**What this is:** a live demo of the Phase 0–4 shadow-mode tooling, running against
**self-seeded synthetic data** (`sfe.scripts.seed_demo_data`) — there is no real Terna
connection (PRD Phase 0 is still open) and no live trading decisions are made anywhere in
this codebase. Say so if you share the link.

## One-time setup

1. **Push this repo to GitHub.** Render deploys from a connected GitHub (or GitLab) repo.
2. In the [Render dashboard](https://dashboard.render.com), **New +** → **Blueprint**, and
   point it at the repo. Render reads `render.yaml` and creates the `sfe-dashboard` web
   service automatically — no manual build/start command entry needed.
3. Optional: on the service's **Environment** tab, set `DASHBOARD_PASSWORD` to require a
   password before the app renders anything (leave unset for an open demo).
4. Deploy. First boot runs `sfe.scripts.seed_demo_data` (≈30–60s: lands ~1 month of
   synthetic Terna/GME data and trains the D-1/MI1/MI2 model set) before Streamlit starts.

## Storage: ephemeral by default

The **free** plan's disk is wiped on every deploy and on restart after idling — the app
reseeds itself each time (that's the point of `seed_demo_data`; it's idempotent and skips
the backfill if data is already present). Two consequences:

- The **Audit & P&L** and **Backtest Runs** tabs start empty after every restart — they
  only fill in as you use the running app (generate + record forecasts, or run
  `train_baseline` / `run_backtest` / `retrain_check` against the same storage root from a
  shell with `SFE_STORAGE__ROOT` pointed at the same path).
- Nothing you record is durable across a redeploy.

To persist landed data, trained models, the audit log and the run log across restarts:
add a [Render Disk](https://render.com/docs/disks) to the service (requires a paid instance
type — Disks aren't available on the free plan) mounted at, say, `/var/data`, and set
`SFE_STORAGE__ROOT=/var/data` as an env var.

## Cost / plan notes

- `plan: free` in `render.yaml` spins the service down after 15 minutes of inactivity; the
  next request wakes it (cold start ≈ seed time + app boot, so up to ~1 minute).
- Swap `plan: free` for `starter` (or higher) in `render.yaml` for an always-on instance
  and disk eligibility.

## Updating

Push to the connected branch; Render redeploys automatically. `render.yaml` changes (plan,
env vars, commands) also apply on the next deploy once pushed.
