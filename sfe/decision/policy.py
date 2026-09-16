"""Decision policy (PRD F-11..F-13): spread distribution + sign probability -> recommended Delta.

Sign convention for ``Delta`` (deliberate deviation = procured volume - load):
  Delta > 0  over-procure, sell the surplus into imbalance  (area short, S > 0)
  Delta < 0  under-procure, settle the shortfall vs Terna    (area long,  S < 0)

Every recommendation is net of the non-arbitrage charge, risk-limited, and may be an
explicit abstention.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from sfe.config import load_yaml
from sfe.curate.timebase import utc_to_rome
from sfe.decision.abstain import AbstainRule
from sfe.decision.charges import ChargeModel
from sfe.decision.limits import Limits
from sfe.decision.sizing import Sizer
from sfe.io.storage import StorageBackend

_QCOL = re.compile(r"^S_p(\d{1,2})$")


@dataclass
class Recommendation:
    area: str
    mtu_start: pd.Timestamp
    sign_prob: float
    e_spread: float
    sigma_spread: float
    charge_eur_per_mwh: float
    load_mwh: float
    direction: int
    delta_mwh: float
    abstain: bool
    reason: str
    expected_net_margin_eur: float
    cvar_eur_per_mwh: float
    limit_binding: str | None


def _quantile_cols(cols) -> dict[float, str]:
    out = {}
    for c in cols:
        m = _QCOL.match(str(c))
        if m:
            out[int(m.group(1)) / 100.0] = c
    return dict(sorted(out.items()))


def _default_load() -> float:
    return float(load_yaml("decision").get("portfolio", {}).get("default_load_mwh", 200.0))


def recommend(
    quantiles: dict[float, float],
    *,
    sign_prob: float,
    charge_eur_per_mwh: float,
    load_mwh: float,
    sizer: Sizer,
    limits: Limits,
    abstain_rule: AbstainRule,
    area: str = "",
    mtu_start: pd.Timestamp | None = None,
) -> Recommendation:
    e_spread = sizer.expected_spread(quantiles)
    sigma = sizer.sigma_spread(quantiles)
    edge = sizer.net_edge(e_spread, charge_eur_per_mwh)
    direction = int(np.sign(e_spread)) or 1

    reason = abstain_rule.check(edge, sigma, sign_prob) or ""
    delta = 0.0 if reason else sizer.target_delta(quantiles, charge_eur_per_mwh)

    binding = None
    if delta != 0.0:
        delta, binding = limits.clip(delta, load_mwh)
        if not limits.cvar_ok(quantiles, charge_eur_per_mwh, direction):
            delta, reason = 0.0, "cvar_limit"

    eff_s = float(sizer.impact.effective_spread(e_spread, delta))
    net_per_mwh = eff_s - direction * charge_eur_per_mwh if delta != 0.0 else 0.0
    return Recommendation(
        area=area,
        mtu_start=mtu_start,
        sign_prob=float(sign_prob),
        e_spread=float(e_spread),
        sigma_spread=float(sigma),
        charge_eur_per_mwh=float(charge_eur_per_mwh),
        load_mwh=float(load_mwh),
        direction=direction,
        delta_mwh=float(delta),
        abstain=bool(reason),
        reason=reason or "sized",
        expected_net_margin_eur=float(delta * net_per_mwh),
        cvar_eur_per_mwh=float(limits.cvar_per_mwh(quantiles, charge_eur_per_mwh, direction)),
        limit_binding=binding,
    )


def recommend_frame(
    df: pd.DataFrame,
    *,
    as_of: object | None = None,
    storage: StorageBackend | None = None,
    charge_model: ChargeModel | None = None,
    sizer: Sizer | None = None,
    limits: Limits | None = None,
    abstain_rule: AbstainRule | None = None,
    sign_prob_col: str = "sign_prob",
    load_col: str = "load_mwh",
) -> pd.DataFrame:
    """Vectorised policy over a frame with ``_area``, ``_mtu_start``, ``S_p*`` and a
    sign-probability column. Applies the per-day aggregate cap after per-MTU sizing.
    """
    charge_model = charge_model or ChargeModel.from_config()
    sizer = sizer or Sizer.from_config()
    limits = limits or Limits.from_config()
    abstain_rule = abstain_rule or AbstainRule.from_config()

    qcols = _quantile_cols(df.columns)
    if not qcols:
        raise ValueError("recommend_frame needs S_p10..S_p90 columns")

    out = df.copy().reset_index(drop=True)
    out["_mtu_start"] = pd.to_datetime(out["_mtu_start"], utc=True)
    charges = charge_model.charge_per_mwh(
        out[["_area", "_mtu_start"]], as_of=as_of, storage=storage
    ).to_numpy()
    load = (
        pd.to_numeric(out[load_col], errors="coerce").fillna(_default_load()).to_numpy()
        if load_col in out.columns
        else np.full(len(out), _default_load())
    )
    sign_prob = (
        pd.to_numeric(out[sign_prob_col], errors="coerce").to_numpy()
        if sign_prob_col in out.columns
        else np.full(len(out), np.nan)
    )

    recs = []
    for i, row in out.iterrows():
        q = {qq: float(row[c]) for qq, c in qcols.items()}
        recs.append(
            recommend(
                q,
                sign_prob=sign_prob[i],
                charge_eur_per_mwh=float(charges[i]),
                load_mwh=float(load[i]),
                sizer=sizer,
                limits=limits,
                abstain_rule=abstain_rule,
                area=row["_area"],
                mtu_start=row["_mtu_start"],
            )
        )
    rec_df = pd.DataFrame([asdict(r) for r in recs])

    # per-day aggregate cap
    day = rec_df["mtu_start"].map(lambda t: utc_to_rome(t.to_pydatetime()).date())
    capped = limits.apply_daily_cap(rec_df["delta_mwh"], group=day)
    scaled = capped.ne(rec_df["delta_mwh"])
    rec_df.loc[scaled, "limit_binding"] = "daily_cap"
    rec_df["delta_mwh"] = capped
    rec_df["expected_net_margin_eur"] = np.where(
        rec_df["delta_mwh"].abs() > 0,
        rec_df["expected_net_margin_eur"]
        * (capped / rec_df["delta_mwh"].replace(0, np.nan)).fillna(0.0),
        0.0,
    )

    rec_df = rec_df.drop(columns=["area", "mtu_start"])
    # rec_df is authoritative for any column it shares with the input frame (e.g. sign_prob,
    # load_mwh) - drop the input copies so the result has no duplicate columns.
    dup = [c for c in rec_df.columns if c in out.columns]
    return pd.concat([out.drop(columns=dup), rec_df], axis=1)
