"""Write a small simulated dataset in fitzRoy's AFL-API column format, for testing the
pipeline end to end without network access. Not used for any reported result.

    python -m tests.make_fake_raw /path/to/fake_project
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(1)
STAT_COLS = {
    "disposals": 18, "kicks": 10, "handballs": 8, "marks": 4.5, "contestedPossessions": 7, "uncontestedPossessions": 11,
    "tackles": 3.5, "inside50s": 2.5, "rebound50s": 2, "clearances.totalClearances": 2.5, "hitouts": 1,
    "goals": .7, "behinds": .5, "scoreInvolvements": 4, "metresGained": 300, "intercepts": 3, "onePercenters": 2,
    "contestedMarks": .8, "marksInside50": .6, "turnovers": 3, "clangers": 3, "goalAssists": .4,
    "extendedStats.pressureActs": 15, "extendedStats.groundBallGets": 4, "ratingPoints": 9,
}


def main(root: Path) -> None:
    raw = root / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    n = 900
    talent = rng.normal(0, 1, n)
    style = rng.dirichlet([1, 1, 1], n)
    first_season = rng.integers(2014, 2026, n)
    clubs = [f"Club {i}" for i in range(18)]
    club = rng.integers(0, 18, n)
    lists = []
    for comp, seasons in [("AFLM", range(2015, 2027)), ("VFL", [2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026])]:
        for s in seasons:
            rows = []
            exp = np.clip(s - first_season, 0, 8)
            ability = talent + .25 * exp + rng.normal(0, .3, n)
            active = (first_season <= s) & (s - first_season < 10)
            for rnd in range(1, 21):
                if comp == "AFLM":
                    pool = np.where(active & (ability + rng.normal(0, .5, n) > .9))[0]
                    lvl = 1.0
                else:
                    pool = np.where(active & (ability + rng.normal(0, .5, n) <= .9) & (ability > -1.2))[0]
                    lvl = 1.3  # VFL output is inflated relative to AFL
                for pid in pool:
                    q = np.exp(.35 * ability[pid]) * lvl
                    r = {"providerId": f"{comp}{s}R{rnd}M{club[pid] // 2}", "utcStartTime": f"{s}-04-{1 + rnd % 27:02d}T03:00:00Z",
                         "round.roundNumber": rnd, "home.team.name": clubs[club[pid] // 2 * 2], "away.team.name": clubs[club[pid] // 2 * 2 + 1],
                         "team.name": clubs[club[pid]] + (" VFL" if comp == "VFL" else ""),
                         "player.player.player.playerId": f"CD_I{100000 + pid}", "player.player.player.givenName": "P",
                         "player.player.player.surname": str(pid), "player.player.position": "INT",
                         "timeOnGroundPercentage": rng.uniform(65, 92)}
                    for k, base in STAT_COLS.items():
                        mult = 1 + (style[pid][0] - .33) * (k in ("clearances.totalClearances", "contestedPossessions"))
                        r[k] = rng.poisson(max(base * q * mult, .05))
                    r["playerStats.lastUpdated"] = "x"
                    rows.append(r)
            df = pd.DataFrame(rows)
            df.to_csv(raw / f"{comp}_{s}.csv.gz", index=False, compression="gzip")
            if comp == "AFLM":
                for pid in np.where(active)[0]:
                    lists.append({"providerId": f"CD_I{100000 + pid}", "team.name": clubs[club[pid]], "list_season": s})
    pd.DataFrame(lists).to_csv(raw / "afl_lists.csv.gz", index=False, compression="gzip")
    print("fake raw written to", raw)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
