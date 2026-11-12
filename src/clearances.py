"""Why clearances shrink the most: role or level?

Takes every player-season where a listed player was a genuine VFL midfielder (3+ clearances a
game) and also played 3+ AFL games that season, then groups them by how many AFL centre-bounce
attendances (CBAs) he got. The VFL feed doesn't record CBAs, so the AFL side is the only one
we can see.

Output: outputs/clearance_roles.csv and a printed summary.
Run:    python -m src.clearances
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW = Path("data/raw")
OUT = Path("outputs")
COLS = {"providerId": "match_id", "team.name": "team", "player.player.player.playerId": "player_id",
        "clearances.totalClearances": "clearances", "extendedStats.centreBounceAttendances": "cba",
        "timeOnGroundPercentage": "tog"}


def load() -> pd.DataFrame:
    frames = []
    for f in sorted(RAW.glob("AFLM_*.csv.gz")) + sorted(RAW.glob("VFL_*.csv.gz")):
        comp, season = f.name.split(".")[0].split("_")
        if int(season) < 2021:  # VFL data starts in 2021
            continue
        d = pd.read_csv(f, low_memory=False)
        d = d[[c for c in COLS if c in d]].rename(columns=COLS)
        d["comp"], d["season"] = ("AFL" if comp == "AFLM" else "VFL"), int(season)
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    g = load()
    team_pg = g.groupby(["comp", "match_id", "team"]).clearances.sum().groupby("comp").mean()  # every player counted
    g = g[~((g.comp == "AFL") & (g.tog < 50))]  # then drop AFL sub/injury games, as elsewhere
    r = g.groupby(["player_id", "comp", "season"]).agg(games=("clearances", "size"), clearances=("clearances", "mean"),
                                                        cba=("cba", "mean")).reset_index()
    v = r[(r.comp == "VFL") & (r.games >= 3)]
    a = r[(r.comp == "AFL") & (r.games >= 3)]
    p = v.merge(a, on=["player_id", "season"], suffixes=("_vfl", "_afl"))
    mids = p[p.clearances_vfl >= 3].copy()
    mids["kept"] = mids.clearances_afl / mids.clearances_vfl
    mids["afl_role"] = pd.cut(mids.cba_afl, [-1, 5, 15, 99], labels=["Fewer than 5 CBAs", "5 to 15 CBAs", "15+ CBAs"])
    out = mids.groupby("afl_role", observed=True).agg(player_seasons=("kept", "size"), vfl_clearances=("clearances_vfl", "mean"),
                                                      afl_clearances=("clearances_afl", "mean"), share_kept=("kept", "mean")).round(2)
    OUT.mkdir(exist_ok=True)
    out.to_csv(OUT / "clearance_roles.csv")
    print("Team clearances per game:", team_pg.round(1).to_dict())
    print(f"VFL midfielders (3+ clearances/game) with 3+ AFL games the same season: {len(mids)} player-seasons")
    print(f"  VFL {mids.clearances_vfl.mean():.1f} -> AFL {mids.clearances_afl.mean():.1f} clearances/game; "
          f"{(mids.cba_afl < 5).mean():.0%} averaged fewer than 5 AFL CBAs")
    print(out.to_string())


if __name__ == "__main__":
    main()
