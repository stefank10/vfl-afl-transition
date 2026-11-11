"""Shared helpers: which stats to use, per-game rates and role groups."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED = Path("data/processed")
OUTPUTS = Path("outputs")

# Stats we try to compare across competitions, with plain-English labels for coaches.
LABELS = {
    "disposals": "Disposals", "kicks": "Kicks", "handballs": "Handballs", "marks": "Marks",
    "contested_poss": "Contested possessions", "uncontested_poss": "Uncontested possessions",
    "tackles": "Tackles", "pressure_acts": "Pressure acts", "inside50s": "Inside 50s",
    "rebound50s": "Rebound 50s", "clearances": "Clearances", "hitouts": "Hitouts", "goals": "Goals",
    "score_involvements": "Score involvements", "metres_gained": "Metres gained", "intercepts": "Intercepts",
    "one_percenters": "One percenters", "contested_marks": "Contested marks", "marks_inside50": "Marks inside 50",
    "ground_ball_gets": "Ground ball gets", "effective_disposals": "Effective disposals",
    "rating_points": "Rating points (CD)",
}
ROLES = ["Key defender", "General defender", "Midfielder", "General forward", "Key forward", "Ruck"]
KEY_POSITION_CM = 194
_HEIGHTS: pd.Series | None = None


def heights() -> pd.Series:
    """Latest recorded height per player from the AFL lists (0 or missing treated as unknown)."""
    global _HEIGHTS
    if _HEIGHTS is None:
        l = load_lists()
        if l is None or "height_cm" not in l:
            _HEIGHTS = pd.Series(dtype=float)
        else:
            l = l[l.height_cm > 150].sort_values("list_season")
            _HEIGHTS = l.groupby("player_id").height_cm.last()
    return _HEIGHTS


def load_games() -> pd.DataFrame:
    g = pd.read_parquet(PROCESSED / "player_games.parquet")
    g["date"] = pd.to_datetime(g["date"])
    return g


def load_lists() -> pd.DataFrame | None:
    f = PROCESSED / "lists.parquet"
    return pd.read_parquet(f) if f.exists() else None


def usable_stats(g: pd.DataFrame, min_coverage: float = 0.9) -> list[str]:
    """Core stats: recorded for at least 90% of rows in BOTH competitions, every season."""
    keep = []
    for s in LABELS:
        if s not in g:
            continue
        cov = g.groupby("comp")[s].apply(lambda x: x.notna().mean())
        if len(cov) == 2 and cov.min() >= min_coverage:
            keep.append(s)
    return keep


def recent_stats(g: pd.DataFrame, since: int = 2024, min_coverage: float = 0.85) -> list[str]:
    """Stats the VFL feed only records from `since` onwards (e.g. contested possessions)."""
    core = set(usable_stats(g))
    v = g[(g.comp == "VFL") & (g.season >= since)]
    a = g[g.comp == "AFL"]
    return [s for s in LABELS if s in g and s not in core
            and v[s].notna().mean() >= min_coverage and a[s].notna().mean() >= 0.9]


def use_tog(g: pd.DataFrame) -> bool:
    """Adjust for time on ground only if both competitions record it."""
    cov = g.groupby("comp")["tog"].apply(lambda x: x.notna().mean())
    return len(cov) == 2 and cov.min() >= 0.9


def rate_frame(g: pd.DataFrame, stats: list[str], tog: bool) -> pd.DataFrame:
    """Per-game values. With TOG, scale to a full game and drop sub/injury games (<50% TOG)."""
    g = g.copy()
    if not tog:
        # VFL has no time on ground, so rates stay per game; AFL sub and injury games (<50% TOG)
        # are still removed so they don't drag the AFL side down
        return g[~((g.comp == "AFL") & (g.tog < 50))]
    if tog:
        g = g[g["tog"] >= 50]
        scale = 100.0 / g["tog"]
        for s in stats:
            g[s] = g[s] * scale
    return g


def assign_roles(rates: pd.DataFrame) -> pd.Series:
    """Role from what the player did, scored within each competition-season so it reflects
    style rather than level: whichever of rebound 50s (defender), clearances (midfielder) or
    goals (forward) he is furthest above average in. Ruck first, on 8+ hitouts a game.
    Defenders and forwards are then split into key position (194cm+) and general by listed height.
    Only stats the VFL records every season are used, so roles mean the same thing in both."""
    def z(col):
        grp = rates.groupby(["comp", "season"])[col]
        return ((rates[col] - grp.transform("mean")) / grp.transform("std")).fillna(0)

    score = pd.DataFrame({"Defender": z("rebound50s"), "Midfielder": z("clearances"), "Forward": z("goals")})
    role = score.idxmax(axis=1)
    # split defenders and forwards by height from the AFL lists: 194cm+ = key position
    if "player_id" in rates:
        tall = rates["player_id"].map(heights()).fillna(0).values >= KEY_POSITION_CM
        role = pd.Series(np.where(role == "Midfielder", "Midfielder",
                                  np.where(tall, "Key " + role.str.lower(), "General " + role.str.lower())),
                         index=rates.index)
    return role.where(rates["hitouts"] < 8, "Ruck")


def season_rates(g: pd.DataFrame, stats: list[str], min_games: int = 1) -> pd.DataFrame:
    agg = g.groupby(["player_id", "comp", "season"]).agg(
        games=("match_id", "size"), player=("player", "last"), team=("team", "last"),
        **{s: (s, "mean") for s in stats})
    agg = agg[agg["games"] >= min_games].reset_index()
    agg["role"] = assign_roles(agg)
    return agg
