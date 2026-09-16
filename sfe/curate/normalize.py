"""Defensive parsing of raw source records (PRD 6.1: "numeric fields returned as strings").

Every value that lands in the store passes through here. The functions never raise on bad
input - they return ``None`` and let the caller decide - because a single malformed field
must not stall a backfill.
"""

from __future__ import annotations

import re
from typing import Any

_THOUSANDS = re.compile(r"(?<=\d)[ '](?=\d)")
_NULLISH = {"", "-", "--", "n/a", "na", "null", "none", "nan", "#n/d", "nd"}

_SIGN_POS = {"+", "1", "+1", "pos", "positive", "long", "surplus", "eccedenza", "a credito"}
_SIGN_NEG = {"-", "-1", "neg", "negative", "short", "deficit", "a debito"}
_SIGN_ZERO = {"0", "flat", "balanced", "nullo"}


def parse_number(value: Any) -> float | None:
    """Parse a possibly-string number. Handles ``,`` decimal comma and thousands marks."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        f = float(value)
        return None if f != f else f  # drop NaN
    s = str(value).strip()
    if s.lower() in _NULLISH:
        return None
    s = _THOUSANDS.sub("", s)
    # If there is both a dot and a comma, assume the last-seen one is the decimal sep.
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def parse_sign(value: Any) -> int | None:
    """Map an imbalance-sign field to ``+1`` (area long), ``-1`` (area short), ``0``."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if value != value:
            return None
        return 1 if value > 0 else (-1 if value < 0 else 0)
    s = str(value).strip().lower()
    # Check the sign vocabulary before the nullish set: "-" is a valid negative marker
    # here even though it reads as "missing" for a numeric field.
    if s in _SIGN_POS:
        return 1
    if s in _SIGN_NEG:
        return -1
    if s in _SIGN_ZERO:
        return 0
    if s in _NULLISH:
        return None
    n = parse_number(s)
    if n is None:
        return None
    return 1 if n > 0 else (-1 if n < 0 else 0)


def sign_from_number(mwh: float | None) -> int | None:
    """Derive the sign from a signed imbalance volume."""
    if mwh is None:
        return None
    return 1 if mwh > 0 else (-1 if mwh < 0 else 0)


def coerce_str(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s or None
