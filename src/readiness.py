"""C. Transition readiness: what did VFL players who went on to establish themselves in the AFL
look like before they debuted, and how quickly did debutants adapt?

Cohort: every player whose AFL debut came in 2019-2024 (so each has at least two seasons of
follow-up), with at least 4 VFL games in the 400 days before debut.
Outcome: "established" = 20+ AFL games across the debut season and the season after.

Outputs
- readiness_effects.csv    how far apart the two groups were on each pre-debut VFL stat
                           (standardised difference, 90% bootstrap interval)
- readiness_model.json     can pre-debut VFL numbers tell the groups apart? (repeated
                           cross-validated AUC, against disposals-only)
- readiness_benchmarks.csv by role: what established debutants averaged in the VFL beforehand
- adaptation.csv           output in AFL games 1-30, relative to each player's translated VFL level
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .common import LABELS, OUTPUTS, assign_roles, load_games, rate_frame, usable_stats, use_tog
from .translate import debuts

ESTABLISHED_GAMES = 20
RNG = np.random.default_rng(32)


def build_cohort(g: pd.DataFrame, stats: list[str]) -> pd.DataFrame:
    deb = debuts(g)
    deb = deb[deb.debut_season <= 2024]
    m = g.merge(deb[["player_id", "debut_date", "debut_season"]], on="player_id")
    pre = m[(m.comp == "VFL") & (m.date < m.debut_date) & (m.date >= m.debut_date - pd.Timedelta(days=400))]
    pre = pre.sort_values("date").groupby("player_id").tail(10)
    # z-score within VFL season so different eras of the competition compare fairly
    for s in stats:
        grp = g[g.comp == "VFL"].groupby("season")[s]
        mu, sd = grp.mean(), grp.std()
        pre[f"{s}_z"] = (pre[s] - pre.season.map(mu)) / pre.season.map(sd)
    feats = pre.groupby("player_id").agg(vfl_games=("match_id", "size"),
                                         **{s: (s, "mean") for s in stats},
                                         **{f"{s}_z": (f"{s}_z", "mean") for s in stats}).reset_index()
    follow = m[(m.comp == "AFL") & (m.season <= m.debut_season + 1)].groupby("player_id").size().rename("afl_games_2yrs")
    c = deb.merge(feats, on="player_id").merge(follow, on="player_id", how="left")
    c = c[c.vfl_games >= 4].copy()
    c["established"] = (c.afl_games_2yrs >= ESTABLISHED_GAMES).astype(int)
    prof = c[["player_id"] + stats].assign(comp="VFL", season=0)
    c["role"] = assign_roles(prof).values
    return c


def effects(c: pd.DataFrame, stats: list[str], n_boot: int = 2000) -> pd.DataFrame:
    def d(df, s):
        a, b = df.loc[df.established == 1, f"{s}_z"], df.loc[df.established == 0, f"{s}_z"]
        sp = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / (len(a) + len(b) - 2))
        return (a.mean() - b.mean()) / sp if sp > 0 else np.nan
    rows = []
    for s in stats:
        boots = [d(c.iloc[RNG.integers(0, len(c), len(c))], s) for _ in range(n_boot)]
        lo, hi = np.nanpercentile(boots, [5, 95])
        rows.append({"stat": s, "label": LABELS[s], "d": d(c, s), "lo": lo, "hi": hi,
                     "established_mean": c.loc[c.established == 1, s].mean(),
                     "other_mean": c.loc[c.established == 0, s].mean()})
    return pd.DataFrame(rows).sort_values("d")


def cv_auc(c: pd.DataFrame, cols: list[str]) -> tuple[float, float, float]:
    X, y = c[cols].fillna(0).values, c.established.values
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=7)
    scores = []
    for tr, te in cv.split(X, y):
        mdl = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000))
        mdl.fit(X[tr], y[tr])
        scores.append(roc_auc_score(y[te], mdl.predict_proba(X[te])[:, 1]))
    return float(np.mean(scores)), float(np.percentile(scores, 5)), float(np.percentile(scores, 95))


def benchmarks(c: pd.DataFrame, stats: list[str]) -> pd.DataFrame:
    rows = []
    for role, df in c.groupby("role"):
        est, oth = df[df.established == 1], df[df.established == 0]
        for s in stats:
            rows.append({"role": role, "stat": s, "label": LABELS[s], "n_established": len(est), "n_other": len(oth),
                         "est_p25": est[s].quantile(.25), "est_median": est[s].median(), "est_p75": est[s].quantile(.75),
                         "other_median": oth[s].median()})
    return pd.DataFrame(rows)


def adaptation(g: pd.DataFrame, c: pd.DataFrame, factors: dict, stat: str) -> pd.DataFrame:
    """For each debutant: AFL output in game n divided by (pre-debut VFL average x factor)."""
    a = g[(g.comp == "AFL")].merge(c[["player_id", "debut_date", stat, "established"]].rename(columns={stat: "pre"}),
                                   on="player_id")
    a = a[a.date >= a.debut_date].sort_values("date")
    a["game_no"] = a.groupby("player_id").cumcount() + 1
    a = a[(a.game_no <= 30) & (a.pre > 0)]
    a["ratio"] = a[stat] / (a.pre * factors[stat])
    a["bin"] = pd.cut(a.game_no, [0, 3, 6, 10, 15, 20, 30], labels=["1-3", "4-6", "7-10", "11-15", "16-20", "21-30"])
    rows = []
    for (b, est), df in a.groupby(["bin", "established"], observed=True):
        per_player = df.groupby("player_id").ratio.mean()
        boots = [RNG.choice(per_player.values, len(per_player)).mean() for _ in range(1000)]
        rows.append({"bin": str(b), "established": int(est), "players": len(per_player), "ratio": per_player.mean(),
                     "lo": np.percentile(boots, 5), "hi": np.percentile(boots, 95)})
    return pd.DataFrame(rows)


def main() -> None:
    g = load_games()
    stats = usable_stats(g)
    g = rate_frame(g, stats, use_tog(g))
    c = build_cohort(g, stats)
    eff = effects(c, stats)
    zcols = [f"{s}_z" for s in stats]
    auc_full = cv_auc(c, zcols)
    auc_disp = cv_auc(c, ["disposals_z"])
    head = "rating_points" if "rating_points" in stats else "disposals"
    auc_head = cv_auc(c, [f"{head}_z"])
    bench = benchmarks(c, stats)
    ft = pd.read_csv(OUTPUTS / "translation_factors.csv")
    factors = dict(zip(ft.stat, ft.factor))
    adapt = adaptation(g, c, factors, head)

    eff.to_csv(OUTPUTS / "readiness_effects.csv", index=False)
    bench.to_csv(OUTPUTS / "readiness_benchmarks.csv", index=False)
    adapt.to_csv(OUTPUTS / "adaptation.csv", index=False)
    c.drop(columns=[x for x in c.columns if x.endswith("_z")]).to_csv(OUTPUTS / "readiness_cohort.csv", index=False)
    model = {"n": int(len(c)), "n_established": int(c.established.sum()), "debut_seasons": "2019-2024",
             "established_rule": f"{ESTABLISHED_GAMES}+ AFL games in debut season and the next",
             "auc_all_stats": auc_full, "auc_disposals_only": auc_disp, "headline_stat": head, "auc_headline": auc_head,
             "by_role": c.groupby("role").agg(n=("established", "size"), rate=("established", "mean")).round(3).reset_index().to_dict("records")}
    (OUTPUTS / "readiness_model.json").write_text(json.dumps(model, indent=2))
    print(json.dumps(model, indent=2))
    print(eff.round(2).to_string(index=False))
    print(adapt.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
