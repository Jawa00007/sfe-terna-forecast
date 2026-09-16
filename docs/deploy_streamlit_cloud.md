# Deploying the dashboard to Streamlit Community Cloud

Free, no card required. Unlike Render, Streamlit Cloud has no CLI/API for creating an
app — the one-time setup below is UI-only by design (their platform, not a limitation on
this repo's side).

**What this is:** a live demo of the Phase 0–4 shadow-mode tooling running on
**self-seeded synthetic data** — there is no real Terna connection (PRD Phase 0 is still
open). Say so if you share the link.

## One-time setup

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. **New app** → **Deploy a public app from GitHub**.
3. Fill in:
   - Repository: your repo (e.g. `Jawa00007/sfe-terna-forecast`)
   - Branch: `master`
   - Main file path: `sfe/dashboard/app.py`
4. **Deploy**. First load takes ~1–2 minutes: the app self-seeds on import (lands ~1 month
   of synthetic Terna/GME data, trains the D-1/MI1/MI2 model set) via `_ensure_seeded()` in
   `sfe/dashboard/app.py`, before rendering anything.
5. Optional password gate: app **Settings → Secrets**, add:
   ```toml
   DASHBOARD_PASSWORD = "whatever-you-want"
   ```
   (the app reads it as an env var — Streamlit Cloud exposes Secrets that way).

Streamlit Cloud reads `requirements.txt` (`-e .[deploy]` — installs this package + the
dashboard/model dependencies) and `runtime.txt` (pins Python 3.11).

## Storage: ephemeral

The container's disk does not persist across a reboot/redeploy/sleep-wake cycle — the app
reseeds itself each time it needs to (that's what `_ensure_seeded()` is for). The **Audit &
P&L** and **Backtest Runs** tabs therefore start empty after every restart; they fill in as
you use the running app.

## Updating

Push to the connected branch; Streamlit Cloud redeploys automatically. To force a rebuild
without a code change, use the app's **Reboot** action in the Streamlit Cloud dashboard.

## Also available: Render

`render.yaml` + the same self-seeding still work for Render (see
[`deploy_render.md`](deploy_render.md)) if you later want a paid, always-on, persistent-disk
deployment — Render's free tier now requires a verified card, which is why this repo
defaults to Streamlit Cloud for the no-card path.
