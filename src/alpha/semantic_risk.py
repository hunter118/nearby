from __future__ import annotations

import re


_HEAD_TO_HEAD = re.compile(r"\bvs\.?\b", re.IGNORECASE)
_SPORTS_COMPETITION = re.compile(
    r"\b(?:finals?|championship|cup|tournament|super bowl|world series|playoffs?)\b",
    re.IGNORECASE,
)
_SPORTS_ACTION = re.compile(r"\b(?:win|winner|champion|qualify|make|reach)\b", re.IGNORECASE)
_SPORTS_LEAGUE = re.compile(
    r"\b(?:nba|wnba|nfl|nhl|mlb|mls|ncaa|uefa|fifa|ufc|pga|atp|wta)\b",
    re.IGNORECASE,
)


def semantic_risk_class(question: str) -> str:
    """Classify event-specific competitive markets using question semantics.

    Head-to-head games and winner-take-all sports competitions have concentrated
    event risk: a strong favorite can still lose its entire binary-token
    notional in one result.  The rule is intentionally transparent and requires
    no fitted labels or future market outcome.
    """

    text = question.strip()
    # Standardized single-game labels are terse (for example, "Ravens vs.
    # Bills") and normally do not end in a question mark.  Requiring that form
    # avoids treating propositions such as "2 Trump vs. Harris debates before
    # election?" as winner-take-all sporting events.
    if _HEAD_TO_HEAD.search(text) and not text.endswith("?"):
        return "competitive_event"
    has_competition = bool(_SPORTS_COMPETITION.search(text))
    has_action = bool(_SPORTS_ACTION.search(text))
    has_league = bool(_SPORTS_LEAGUE.search(text))
    if has_action and (has_competition or has_league):
        return "competitive_event"
    return "standard"
