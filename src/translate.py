"""A. AFL-equivalent: translate VFL output into AFL terms.

Method
1. Pairs: every player-season with at least MIN_GAMES games in BOTH the VFL and the AFL.
2. Translation factor per stat = weighted AFL rate / weighted VFL rate across pairs, weighted by
   the harmonic mean of the two game counts (a player with 3 and 15 games counts as about 5).
   90% intervals by bootstrapping players.
3. Validation on debutants the factors never saw: factors fitted on seasons up to 2023, then used
   to predict how 2024-26 debutants went in their first AFL games from their VFL games just
   before debut. Compared with two simpler guesses: raw VFL numbers, and "a typical debutant
   in his role".
   A fourth estimate blends the translated VFL form with the typical debutant in that role,
   weighted by VFL games (a few big games are partly luck); the blend weight is chosen on
   the training debutants only.
4. AFL-equivalent for every AFL-listed player in the latest VFL season, using that validated
   blend, placed against AFL players in the same role (8+ games that season).

Caveat on selection: players are promoted after good VFL weeks and dropped after poor AFL ones,
which makes VFL form look a little better than it is. The adjacent-season version (VFL in year t,
AFL in year t+1) is reported alongside as a check.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .common import LABELS, OUTPUTS, ROLES, load_games, load_lists, rate_frame, recent_stats, season_rates, usable_stats, use_tog

MIN_GAMES = 3
RNG = np.random.default_rng(2026)
K_GRID = [0, 1, 2, 3, 4, 6, 8, 10, 15, 20, 30]


def pairs_same_season(sr: pd.DataFrame) -> pd.DataFrame:
    v = sr[(sr.comp == "VFL") & (sr.games >= MIN_GAMES)]
    a = sr[(sr.comp == "AFL") & (sr.games >= MIN_GAMES)]
    p = v.merge(a, on=["player_id", "season"], suffixes=("_v", "_a"))
    p["w"] = 2 * p.games_v * p.games_a / (p.games_v + p.games_a)
    return p


def pairs_adjacent(sr: pd.DataFrame) -> pd.DataFrame:
    v = sr[(sr.comp == "VFL") & (sr.games >= MIN_GAMES)].copy()
    a = sr[(sr.comp == "AFL") & (sr.games >= MIN_GAMES)].copy()
    a["season"] = a["season"] - 1
    p = v.merge(a, on=["player_id", "season"], suffixes=("_v", "_a"))
    p["w"] = 2 * p.games_v * p.games_a / (p.games_v + p.games_a)
    return p


def factor(p: pd.DataFrame, s: str) -> float:
    p = p[p[f"{s}_a"].notna() & p[f"{s}_v"].notna()]
    num = (p.w * p[f"{s}_a"]).sum()
    den = (p.w * p[f"{s}_v"]).sum()
    return float(num / den) if den > 0 else np.nan


def factors_with_ci(p: pd.DataFrame, stats: list[str], n_boot: int = 1000) -> pd.DataFrame:
    players = p.player_id.unique()
    idx = {pid: np.where(p.player_id.values == pid)[0] for pid in players}
    boots = {s: [] for s in stats}
    for _ in range(n_boot):
        pick = RNG.choice(players, size=len(players), replace=True)
        rows = np.concatenate([idx[x] for x in pick])
        b = p.iloc[rows]
        for s in stats:
            boots[s].append(factor(b, s))
    out = []
    for s in stats:
        lo, hi = np.nanpercentile(boots[s], [5, 95])
        n = int((p[f"{s}_a"].notna() & p[f"{s}_v"].notna()).sum())
        out.append({"stat": s, "label": LABELS[s], "factor": factor(p, s), "lo": lo, "hi": hi, "n_pairs": n})
    return pd.DataFrame(out).sort_values("factor")


def debuts(g: pd.DataFrame) -> pd.DataFrame:
    """First AFL game for each player whose first AFL game in the data is 2019 or later
    (the data starts in 2015, so earlier debutants are excluded rather than misdated)."""
    a = g[g.comp == "AFL"].sort_values("date")
    first = a.groupby("player_id").agg(debut_date=("date", "first"), debut_season=("season", "first"),
                                       player=("player", "first"), afl_team=("team", "first")).reset_index()
    return first[first.debut_season >= 2019]


def window_rates(g: pd.DataFrame, deb: pd.DataFrame, stats: list[str], n_vfl: int = 10, n_afl: int = 10) -> pd.DataFrame:
    """Each debutant's last n_vfl VFL games before debut, and first n_afl AFL games."""
    m = g.merge(deb[["player_id", "debut_date"]], on="player_id")
    pre = m[(m.comp == "VFL") & (m.date < m.debut_date) & (m.date >= m.debut_date - pd.Timedelta(days=400))]
    pre = pre.sort_values("date").groupby("player_id").tail(n_vfl)
    post = m[(m.comp == "AFL") & (m.date >= m.debut_date)].sort_values("date").groupby("player_id").head(n_afl)
    pv = pre.groupby("player_id").agg(vfl_games=("match_id", "size"), **{s: (s, "mean") for s in stats})
    pa = post.groupby("player_id").agg(afl_games=("match_id", "size"), **{s: (s, "mean") for s in stats})
    return pv.join(pa, lsuffix="_v", rsuffix="_a", how="inner").reset_index()


def validate(g: pd.DataFrame, sr: pd.DataFrame, stats: list[str]) -> pd.DataFrame:
    train_sr = sr[sr.season <= 2023]
    f = {s: factor(pairs_same_season(train_sr), s) for s in stats}
    deb = debuts(g)
    w = window_rates(g, deb, stats)
    w = w[(w.vfl_games >= 4) & (w.afl_games >= 4)].merge(deb[["player_id", "debut_season"]], on="player_id")
    # role from the pre-debut VFL profile
    prof = w[["player_id"] + [f"{s}_v" for s in stats]].rename(columns=lambda c: c[:-2] if c.endswith("_v") else c)
    prof["comp"], prof["season"] = "VFL", 0
    from .common import assign_roles
    w["role"] = assign_roles(prof).values
    train, test = w[w.debut_season <= 2023], w[w.debut_season >= 2024]
    rows, ks = [], {}
    for s in stats:
        actual = test[f"{s}_a"]
        sd = w[f"{s}_a"].std()
        typ_tr = train.role.map(train.groupby("role")[f"{s}_a"].mean())
        typical = test.role.map(train.groupby("role")[f"{s}_a"].mean()).fillna(train[f"{s}_a"].mean())
        # blend: translated VFL form, pulled towards the typical debutant by k / (VFL games + k);
        # k is chosen on the training debutants only
        def blend(df, typ, k):
            return (df.vfl_games * df[f"{s}_v"] * f[s] + k * typ) / (df.vfl_games + k)
        k = min(K_GRID, key=lambda k: (blend(train, typ_tr, k) - train[f"{s}_a"]).abs().mean())
        ks[s] = k
        e_raw = (test[f"{s}_v"] - actual).abs().mean()
        e_typ = (typical - actual).abs().mean()
        e_tr = (test[f"{s}_v"] * f[s] - actual).abs().mean()
        e_bl = (blend(test, typical, k) - actual).abs().mean()
        rows.append({"stat": s, "label": LABELS[s], "n_train": len(train), "n_test": len(test), "mae_raw_vfl": e_raw,
                     "mae_typical": e_typ, "mae_translated": e_tr, "mae_blended": e_bl, "k": k, "sd": sd})
    v = pd.DataFrame(rows)
    v["gain_vs_raw"] = 1 - v.mae_blended / v.mae_raw_vfl
    v["gain_vs_typical"] = 1 - v.mae_blended / v.mae_typical
    v["translated_gain_vs_raw"] = 1 - v.mae_translated / v.mae_raw_vfl
    # typical first-10-game AFL output for a debutant in each role, from all debutants (for the live tool)
    typical_all = w.groupby("role")[[f"{s}_a" for s in stats]].mean().rename(columns=lambda c: c[:-2])
    return v, ks, typical_all


def afl_equivalents(g: pd.DataFrame, sr: pd.DataFrame, f: dict, stats: list[str], season: int,
                    ks: dict, typical: pd.DataFrame) -> list[dict]:
    lists = load_lists()
    vg = g[(g.comp == "VFL") & (g.season == season)]
    if lists is not None:
        listed = lists[lists.list_season == season][["player_id", "afl_club"]]
    else:  # fall back: anyone who has played AFL by the end of this season
        ever = g[(g.comp == "AFL") & (g.season <= season)].groupby("player_id").team.last().rename("afl_club")
        listed = ever.reset_index()
    vg = vg.merge(listed, on="player_id")
    per = vg.groupby("player_id").agg(games=("match_id", "size"), player=("player", "last"), vfl_team=("team", "last"),
                                      afl_club=("afl_club", "last"), **{s: (s, "mean") for s in stats}).reset_index()
    per = per[per.games >= 4]
    if per.empty:
        return []
    prof = per.assign(comp="VFL", season=season)
    from .common import assign_roles
    per["role"] = assign_roles(prof).values
    afl = sr[(sr.comp == "AFL") & (sr.season == season) & (sr.games >= 8)]
    afl_roles = afl.groupby("role")
    out = []
    for _, r in per.iterrows():
        bench = afl_roles.get_group(r.role) if r.role in afl_roles.groups else afl
        stats_out = {}
        for s in stats:
            typ = typical.loc[r.role, s] if r.role in typical.index else typical[s].mean()
            eq = (r.games * r[s] * f[s] + ks[s] * typ) / (r.games + ks[s])
            pct = float((bench[s] < eq).mean() * 100)
            stats_out[s] = {"vfl": round(float(r[s]), 2), "eq": round(float(eq), 2),
                            "afl_median": round(float(bench[s].median()), 2), "pct": round(pct)}
        out.append({"id": r.player_id, "player": r.player, "afl_club": r.afl_club, "vfl_team": r.vfl_team,
                    "games": int(r.games), "role": r.role, "stats": stats_out})
    return sorted(out, key=lambda d: (d["afl_club"] or "", d["player"]))


def main() -> None:
    g = load_games()
    stats = usable_stats(g)
    recent = recent_stats(g)
    allst = stats + recent
    tog = use_tog(g)
    g = rate_frame(g, allst, tog)
    sr = season_rates(g, allst)
    same = pairs_same_season(sr)
    adj = pairs_adjacent(sr)
    ft = factors_with_ci(same, allst)
    fa = pd.DataFrame({"stat": allst, "factor_adjacent": [factor(adj, s) for s in allst]})
    ft = ft.merge(fa, on="stat")
    ft["tested"] = ft.stat.isin(stats)
    val, ks, typical = validate(g, sr, stats)
    # recent stats can't be tested out of sample (no VFL record before 2024): use the median
    # blend weight from the tested stats and a typical-debutant value from all debutants
    kmed = float(np.median(list(ks.values())))
    deb = debuts(g)
    w = window_rates(g, deb, allst)
    w = w[(w.vfl_games >= 4) & (w.afl_games >= 4)]
    prof = w[["player_id"] + [f"{s}_v" for s in stats]].rename(columns=lambda c: c[:-2] if c.endswith("_v") else c)
    prof["comp"], prof["season"] = "VFL", 0
    from .common import assign_roles
    w["role"] = assign_roles(prof).values
    for s in recent:
        ks[s] = kmed
        typical[s] = w.groupby("role")[f"{s}_a"].mean()
    season = int(g[g.comp == "VFL"].season.max())
    f = dict(zip(ft.stat, ft.factor))
    eq = afl_equivalents(g, sr, f, allst, season, ks, typical)

    OUTPUTS.mkdir(exist_ok=True)
    ft.to_csv(OUTPUTS / "translation_factors.csv", index=False)
    val.to_csv(OUTPUTS / "translation_validation.csv", index=False)
    summary = {"tog_adjusted": bool(tog), "stats": allst, "core_stats": stats, "recent_stats": recent, "n_pairs": int(len(same)), "n_players": int(same.player_id.nunique()),
               "n_pairs_adjacent": int(len(adj)), "seasons": sorted(map(int, sr.season.unique())),
               "eq_season": season, "n_listed_vfl": len(eq),
               "validation_n": int(val.n_test.iloc[0]) if len(val) else 0,
               "median_gain_vs_raw": float(val.gain_vs_raw.median()), "median_gain_vs_typical": float(val.gain_vs_typical.median()),
               "median_translated_gain_vs_raw": float(val.translated_gain_vs_raw.median()),
               "validation_n_train": int(val.n_train.iloc[0]) if len(val) else 0}
    (OUTPUTS / "translation_summary.json").write_text(json.dumps(summary, indent=2))
    (OUTPUTS / "afl_equivalents.json").write_text(json.dumps(eq))
    print(json.dumps(summary, indent=2))
    print(ft.round(3).to_string(index=False))
    print(val.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
