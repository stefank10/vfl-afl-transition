"""Load the raw fitzRoy pulls into one tidy player-match table.

fitzRoy's AFL-API output has long dotted column names (e.g. player.player.player.playerId,
extendedStats.pressureActs) that have shifted between package versions, so every column
is found by its suffix rather than its full name.

Output: data/processed/player_games.parquet  one row per player per game, AFL and VFL
        data/processed/lists.parquet         AFL list membership by season
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw")
OUT = Path("data/processed")

# tidy name -> column-name suffixes to try, in order
STATS = {
    "disposals": ["disposals"], "kicks": ["kicks"], "handballs": ["handballs"], "marks": ["marks"],
    "contested_poss": ["contestedPossessions"], "uncontested_poss": ["uncontestedPossessions"],
    "tackles": ["tackles"], "inside50s": ["inside50s"], "rebound50s": ["rebound50s"],
    "clearances": ["clearances.totalClearances", "totalClearances"], "hitouts": ["hitouts"],
    "goals": ["goals"], "behinds": ["behinds"], "score_involvements": ["scoreInvolvements"],
    "metres_gained": ["metresGained"], "intercepts": ["intercepts"], "one_percenters": ["onePercenters"],
    "contested_marks": ["contestedMarks"], "marks_inside50": ["marksInside50"],
    "turnovers": ["turnovers"], "clangers": ["clangers"], "goal_assists": ["goalAssists"],
    "pressure_acts": ["extendedStats.pressureActs", "pressureActs"],
    "ground_ball_gets": ["extendedStats.groundBallGets", "groundBallGets"],
    "effective_disposals": ["extendedStats.effectiveDisposals", "effectiveDisposals"],
    "rating_points": ["ratingPoints"],
}
META = {
    "match_id": ["providerId"], "start": ["utcStartTime"], "round": ["round.roundNumber"],
    "home": ["home.team.name"], "away": ["away.team.name"], "team": ["team.name"],
    "player_id": ["player.player.player.playerId", "player.playerId", "playerId"],
    "given": ["player.player.player.givenName", "givenName"],
    "surname": ["player.player.player.surname", "surname"],
    "position": ["player.player.position", "player.position", "position"],
    "tog": ["timeOnGroundPercentage"],
}


def find(cols: list[str], suffixes: list[str]) -> str | None:
    """Exact name first (so "team.name" never resolves to "home.team.name"), then by suffix."""
    for s in suffixes:
        if s in cols:
            return s
    for s in suffixes:
        for c in cols:
            if c.endswith("." + s):
                return c
    return None


def tidy_file(path: Path) -> pd.DataFrame:
    comp, season = re.match(r"(AFLM|VFL)_(\d{4})", path.name).groups()
    df = pd.read_csv(path, low_memory=False)
    cols = list(df.columns)
    out = pd.DataFrame(index=df.index)
    for k, suf in {**META, **STATS}.items():
        c = find(cols, suf)
        out[k] = df[c] if c else np.nan
    out["comp"] = "AFL" if comp == "AFLM" else "VFL"
    out["season"] = int(season)
    out["date"] = pd.to_datetime(out["start"], utc=True, errors="coerce").dt.tz_convert("Australia/Melbourne").dt.date
    out["opponent"] = np.where(out["team"] == out["home"], out["away"], out["home"])
    out["player"] = (out["given"].fillna("").astype(str) + " " + out["surname"].fillna("").astype(str)).str.strip()
    out = out[out["player_id"].notna() & out["disposals"].notna()]
    return out.drop(columns=["start", "given", "surname", "home", "away"])


def load_lists() -> pd.DataFrame | None:
    f = RAW / "afl_lists.csv.gz"
    if not f.exists():
        return None
    df = pd.read_csv(f, low_memory=False)
    # the player ID column is whichever one holds Champion Data IDs (CD_I...)
    idc = next((c for c in df.columns if df[c].astype(str).str.match(r"^CD_I\d+").mean() > 0.9), None)
    teamc = find(list(df.columns), ["team.name", "team"]) or next((c for c in df.columns if "team" in c.lower()), None)
    dobc = find(list(df.columns), ["dateOfBirth"])
    debc = find(list(df.columns), ["debutYear"])
    htc = find(list(df.columns), ["heightInCm"])
    if idc is None:
        return None
    out = pd.DataFrame({"player_id": df[idc], "list_season": df["list_season"],
                        "afl_club": df[teamc] if teamc else np.nan,
                        "dob": pd.to_datetime(df[dobc], errors="coerce") if dobc else pd.NaT,
                        "debut_year": pd.to_numeric(df[debc], errors="coerce") if debc else np.nan,
                        "height_cm": pd.to_numeric(df[htc], errors="coerce") if htc else np.nan})
    return out.drop_duplicates(["player_id", "list_season"])


def main() -> None:
    files = sorted(RAW.glob("AFLM_*.csv.gz")) + sorted(RAW.glob("VFL_*.csv.gz"))
    if not files:
        raise SystemExit("No raw files in data/raw. Run Rscript R/pull_data.R first.")
    games = pd.concat([tidy_file(f) for f in files], ignore_index=True)
    # Integrity: one row per player per match
    dup = games.duplicated(["match_id", "player_id"]).sum()
    games = games.drop_duplicates(["match_id", "player_id"])
    OUT.mkdir(parents=True, exist_ok=True)
    games.to_parquet(OUT / "player_games.parquet", index=False)
    print(f"player-games: {len(games):,} ({dup} duplicate rows dropped)")
    print(games.groupby(["comp", "season"]).agg(rows=("player_id", "size"), players=("player_id", "nunique"),
                                                 tog_present=("tog", lambda s: s.notna().mean()),
                                                 pressure_present=("pressure_acts", lambda s: s.notna().mean())).to_string())
    lists = load_lists()
    if lists is not None:
        lists.to_parquet(OUT / "lists.parquet", index=False)
        print(f"list rows: {len(lists):,}")
    else:
        print("No AFL list file found; listed status will be inferred from AFL games played.")


if __name__ == "__main__":
    main()
