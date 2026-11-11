"""Charts for the README and the web page, plus the JSON the page reads."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .common import OUTPUTS

BROWN, GOLD, INK, SOFT, RULE = "#4D2004", "#C98E00", "#2B1A0E", "#6A5848", "#E3D9CC"
FIG = OUTPUTS / "figures"
CARD_STATS = {"rating_points", "disposals", "contested_poss", "uncontested_poss", "marks", "tackles", "pressure_acts",
              "clearances", "inside50s", "rebound50s", "intercepts", "one_percenters", "score_involvements",
              "metres_gained", "goals", "marks_inside50", "contested_marks", "hitouts"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": RULE, "axes.labelcolor": SOFT,
                     "xtick.color": SOFT, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.spines.left": False, "figure.dpi": 100})


def dot_interval(df: pd.DataFrame, val: str, lo: str, hi: str, ref: float, xlabel: str, title: str, out: Path,
                 fmt=lambda v: f"{v:.2f}", note: str | None = None) -> None:
    df = df.sort_values(val)
    fig, ax = plt.subplots(figsize=(8, .36 * len(df) + 1.4))
    y = range(len(df))
    ax.axvline(ref, color=SOFT, lw=1, ls=(0, (3, 3)))
    ax.hlines(y, df[lo], df[hi], color=GOLD, lw=2.2)
    tested = df["tested"].values if "tested" in df else [True] * len(df)
    for yi, (v, t) in enumerate(zip(df[val], tested)):
        ax.plot(v, yi, "o", color=BROWN if t else "white", ms=7, mec=BROWN if not t else "white", mew=1.5)
    ax.set_yticks(list(y), df["label"])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=RULE, lw=.8)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel)
    # values in a fixed column right of the plot, so they never collide with marks or lines
    for yi, v in enumerate(df[val]):
        ax.text(1.01, yi, fmt(v), transform=ax.get_yaxis_transform(), va="center", fontsize=9, color=SOFT)
    if note:
        fig.text(0.01, 0.01, note, fontsize=8.5, color=SOFT)
    ax.set_title(title, loc="left", fontsize=13, color=INK, fontweight="bold", pad=12)
    fig.tight_layout(rect=(0, .03 if note else 0, .97, 1))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def adaptation_chart(a: pd.DataFrame, stat_label: str, out: Path) -> None:
    order = ["1-3", "4-6", "7-10", "11-15", "16-20", "21-30"]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.axhline(1, color=SOFT, lw=1, ls=(0, (3, 3)))
    ax.text(-0.35, 1.012, "translated VFL level", va="bottom", fontsize=9, color=SOFT)
    for est, col, name in [(1, BROWN, "Went on to establish (20+ games)"), (0, GOLD, "Did not establish")]:
        d = a[a.established == est].set_index("bin").reindex(order).reset_index()
        x = range(len(d))
        ax.fill_between(x, d.lo, d.hi, color=col, alpha=.15, lw=0)
        ax.plot(x, d.ratio, "-o", color=col, lw=2, ms=6, mec="white", mew=1.2, label=name)
    ax.set_xticks(range(len(order)), [f"Games {o}" for o in order], fontsize=9)
    ax.set_ylabel(f"{stat_label}: AFL ÷ translated VFL")
    ax.grid(axis="y", color=RULE, lw=.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.set_title("How quickly debutants reach their translated VFL level", loc="left", fontsize=13, color=INK,
                 fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def recs(df: pd.DataFrame) -> list[dict]:
    """Records with NaN as null: NaN isn't valid JSON and browsers reject the whole file."""
    return json.loads(df.round(3).to_json(orient="records"))


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    ft = pd.read_csv(OUTPUTS / "translation_factors.csv")
    dot_interval(ft, "factor", "lo", "hi", 1.0, "AFL output per unit of VFL output (1.0 = no change)",
                 "What VFL numbers are worth at AFL level", FIG / "translation.png",
                 note="Hollow dots: the VFL feed only records these from 2024, so they aren't tested on unseen debutants.")
    eff = pd.read_csv(OUTPUTS / "readiness_effects.csv")
    dot_interval(eff, "d", "lo", "hi", 0.0, "Standardised difference before debut (0 = no difference)",
                 "Before debut, the VFL numbers barely separated them", FIG / "readiness_effects.png")
    model = json.loads((OUTPUTS / "readiness_model.json").read_text())
    from .common import LABELS
    adaptation_chart(pd.read_csv(OUTPUTS / "adaptation.csv"), LABELS[model["headline_stat"]], FIG / "adaptation.png")

    site = {
        "translation": {"summary": json.loads((OUTPUTS / "translation_summary.json").read_text()),
                        "factors": recs(ft),
                        "validation": recs(pd.read_csv(OUTPUTS / "translation_validation.csv"))},
        "readiness": {"model": model, "effects": recs(eff),
                      "benchmarks": recs(pd.read_csv(OUTPUTS / "readiness_benchmarks.csv"))},
        "players": [{**pl, "stats": {k: v for k, v in pl["stats"].items() if k in CARD_STATS}}
                    for pl in json.loads((OUTPUTS / "afl_equivalents.json").read_text())],
        "labels": LABELS,
    }
    (OUTPUTS / "site_data.json").write_text(json.dumps(site, separators=(",", ":"), allow_nan=False))
    print("figures and site_data.json written")


if __name__ == "__main__":
    main()
