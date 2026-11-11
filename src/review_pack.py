"""Turn a coded VFL game into one-page development review packs, one per player.

Input:  a Sportscode-style XML export (from Hudl Sportscode, or from the browser
        code window in this project) plus a themes file agreed with the
        development coaches.
Output: outputs/packs/<match>/<player>.html   a printable one-page pack
        outputs/packs/<match>/index.html      contents page for the coach
        outputs/development_log.csv           running record of every theme, every game

Every clip in a pack is a link that opens the match vision at that moment
(YouTube ?t= deep link), so no footage is downloaded or re-hosted.

Run:
    python -m src.review_pack --codes examples/demo_codes.xml --themes config/themes_example.json
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

GOOD = {"Kick Effective", "Handball Effective", "Contest Won", "Ground Ball Get", "Mark",
        "Tackle", "Spoil / 1%", "Positioning Good"}
BAD = {"Kick Ineffective", "Handball Ineffective", "Contest Lost", "Turnover",
       "Positioning Poor", "Free Against"}


@dataclass
class Instance:
    id: int
    start: float
    end: float
    code: str
    labels: dict = field(default_factory=dict)

    @property
    def player(self) -> str:
        return self.labels.get("Player", "Unknown")

    @property
    def quarter(self) -> str:
        return self.labels.get("Quarter", "")

    @property
    def outcome(self) -> str:
        if "Outcome" in self.labels:
            return self.labels["Outcome"]
        return "good" if self.code in GOOD else "bad" if self.code in BAD else "neutral"


def read_xml(path: Path) -> tuple[dict, list[Instance]]:
    """Parse a Sportscode-style XML. Tolerates real Sportscode exports, where
    labels can repeat a group and SESSION_INFO may be missing."""
    root = ET.parse(path).getroot()
    info = {}
    si = root.find("SESSION_INFO")
    if si is not None:
        info = {c.tag: (c.text or "").strip() for c in si}
    out = []
    for inst in root.iter("instance"):
        labels = {}
        for lab in inst.findall("label"):
            g = (lab.findtext("group") or "Label").strip()
            t = (lab.findtext("text") or "").strip()
            labels[g] = f"{labels[g]}; {t}" if g in labels else t
        out.append(Instance(int(float(inst.findtext("ID") or 0)), float(inst.findtext("start") or 0),
                            float(inst.findtext("end") or 0), (inst.findtext("code") or "").strip(), labels))
    out.sort(key=lambda i: i.start)
    return info, out


def video_id(url: str) -> str | None:
    m = re.search(r"(?:v=|youtu\.be/|embed/|live/)([A-Za-z0-9_-]{11})", url or "")
    return m.group(1) if m else None


def clip_link(vid: str | None, start: float) -> str | None:
    return f"https://www.youtube.com/watch?v={vid}&t={int(start)}s" if vid else None


def mmss(t: float) -> str:
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


def score_theme(theme: dict, mine: list[Instance]) -> dict:
    """A theme is either a rate (good / (good + bad)) or a count, against a target."""
    if "count" in theme:
        hits = [i for i in mine if i.code in theme["count"]]
        value, target = len(hits), theme.get("target_count")
        return {"kind": "count", "value": value, "target": target, "n": value,
                "met": None if target is None else value >= target, "clips": hits,
                "display": f"{value}", "target_display": f"{target}" if target is not None else "–"}
    good = [i for i in mine if i.code in theme.get("good", [])]
    bad = [i for i in mine if i.code in theme.get("bad", [])]
    n = len(good) + len(bad)
    value = len(good) / n if n else None
    target = theme.get("target")
    return {"kind": "rate", "value": value, "target": target, "n": n,
            "met": None if (value is None or target is None) else value >= target,
            "clips": sorted(good + bad, key=lambda i: i.start),
            "display": "–" if value is None else f"{value:.0%}",
            "target_display": "–" if target is None else f"{target:.0%}"}


CSS = """
:root{--brown:#4D2004;--gold:#FBBF15;--ink:#2B1A0E;--soft:#6A5848;--rule:#E3D9CC;--mist:#F6F1EA;--good:#2E7D4F;--bad:#B23A2E;--neutral:#5A6B85;
--display:"Barlow Condensed","Arial Narrow",sans-serif;--body:"Source Serif 4",Georgia,serif}
*{box-sizing:border-box}body{margin:0;background:#fff;color:var(--ink);font-family:var(--body);font-size:15px;line-height:1.45}
.page{width:min(800px,100% - 32px);margin:24px auto}
header{background:var(--brown);color:#FBF4E8;padding:18px 22px;border-radius:6px 6px 0 0}
header h1{font-family:var(--display);font-size:34px;line-height:1;margin:0}header p{margin:4px 0 0;color:#E9DCC8}
.bar{height:6px;background:var(--gold);border-radius:0 0 6px 6px}
h2{font-family:var(--display);font-weight:600;font-size:22px;margin:22px 0 8px}
.tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:16px}
.tiles div{background:var(--mist);border-radius:6px;padding:10px 12px}.tiles b{display:block;font-family:var(--display);font-size:30px;line-height:1}
.tiles span{font-size:13px;color:var(--soft)}
.theme{border:1px solid var(--rule);border-left:6px solid var(--neutral);border-radius:6px;padding:12px 14px;margin-bottom:10px}
.theme.met{border-left-color:var(--good)}.theme.miss{border-left-color:var(--bad)}
.theme h3{font-family:var(--display);font-size:20px;margin:0 0 4px;display:flex;justify-content:space-between;gap:12px}
.theme .v{font-variant-numeric:tabular-nums}.theme p{margin:0;color:var(--soft);font-size:14px}
ol.clips{margin:8px 0 0;padding-left:20px;columns:2;font-size:14px}ol.clips li{break-inside:avoid}
.g{color:var(--good)}.b{color:var(--bad)}.n{color:var(--neutral)}
table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;padding:4px 6px;border-bottom:1px solid var(--rule)}
th{font-family:var(--display);font-size:16px}td.num{text-align:right;font-variant-numeric:tabular-nums}
.notes{min-height:70px;border:1px dashed var(--rule);border-radius:6px;padding:8px 10px;color:var(--soft)}
footer{margin-top:18px;font-size:12px;color:var(--soft)}
a{color:inherit;text-underline-offset:2px}
@media print{.page{width:auto;margin:0}a{text-decoration:none}}
@media (max-width:600px){.tiles{grid-template-columns:repeat(2,1fr)}ol.clips{columns:1}}
"""
FONTS = ('<link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700&'
         'family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap" rel="stylesheet">')


def quarter_svg(mine: list[Instance]) -> str:
    """Small stacked bars: effective / ineffective / neutral involvements by quarter."""
    qs = ["Q1", "Q2", "Q3", "Q4"]
    counts = {q: Counter(i.outcome for i in mine if i.quarter == q) for q in qs}
    top = max([sum(c.values()) for c in counts.values()] + [1])
    w, h, bw = 360, 120, 56
    parts = [f'<svg viewBox="0 0 {w} {h + 22}" width="100%" style="max-width:{w}px" role="img" '
             f'aria-label="Involvements by quarter">']
    colours = [("good", "#2E7D4F"), ("neutral", "#5A6B85"), ("bad", "#B23A2E")]
    for k, q in enumerate(qs):
        x, y = 20 + k * 85, h
        for key, col in colours:
            v = counts[q][key]
            if not v:
                continue
            bh = v / top * (h - 16)
            y -= bh
            parts.append(f'<rect x="{x}" y="{y:.1f}" width="{bw}" height="{bh:.1f}" fill="{col}" rx="2"/>')
        total = sum(counts[q].values())
        parts.append(f'<text x="{x + bw / 2}" y="{y - 4:.1f}" text-anchor="middle" font-size="12" '
                     f'fill="#2B1A0E">{total}</text>')
        parts.append(f'<text x="{x + bw / 2}" y="{h + 16}" text-anchor="middle" font-size="13" '
                     f'font-family="Barlow Condensed, sans-serif" fill="#6A5848">{q}</text>')
    parts.append("</svg>")
    return "".join(parts)


def clip_list(items: list[Instance], vid: str | None) -> str:
    li = []
    for i in items:
        cls = {"good": "g", "bad": "b"}.get(i.outcome, "n")
        label = f'{i.quarter} {mmss(i.start)} · {html.escape(i.code)}'
        note = f' — <em>{html.escape(i.labels["Note"])}</em>' if i.labels.get("Note") else ""
        link = clip_link(vid, i.start)
        body = f'<a href="{link}" target="_blank" rel="noopener">{label}</a>' if link else label
        li.append(f'<li class="{cls}">{body}{note}</li>')
    return f'<ol class="clips">{"".join(li)}</ol>' if li else "<p>No clips coded for this theme.</p>"


def render_pack(player: str, mine: list[Instance], themes: list[dict], match: str, vid: str | None,
                demo: bool) -> tuple[str, list[dict]]:
    good = sum(i.outcome == "good" for i in mine)
    bad = sum(i.outcome == "bad" for i in mine)
    eff = f"{good / (good + bad):.0%}" if good + bad else "–"
    pressure = sum(i.code in ("Pressure Act", "Tackle") for i in mine)
    cw = sum(i.code == "Contest Won" for i in mine)
    cl = sum(i.code == "Contest Lost" for i in mine)
    contest = f"{cw / (cw + cl):.0%}" if cw + cl else "–"

    theme_html, log_rows = [], []
    for t in themes:
        s = score_theme(t, mine)
        state = "met" if s["met"] else "miss" if s["met"] is False else ""
        verdict = "On target" if s["met"] else "Below target" if s["met"] is False else "Not enough instances"
        n_txt = f" from {s['n']} instances" if s["kind"] == "rate" else ""
        theme_html.append(
            f'<div class="theme {state}"><h3><span>{html.escape(t["name"])}</span>'
            f'<span class="v">{s["display"]} <small>/ target {s["target_display"]}</small></span></h3>'
            f'<p>{verdict}{n_txt}. Clips in order:</p>{clip_list(s["clips"], vid)}</div>')
        log_rows.append({"match": match, "player": player, "theme": t["name"], "kind": s["kind"],
                         "value": "" if s["value"] is None else round(s["value"], 3),
                         "target": "" if s["target"] is None else s["target"], "instances": s["n"],
                         "met": "" if s["met"] is None else s["met"]})

    by_code = Counter(i.code for i in mine)
    code_rows = "".join(f'<tr><td>{html.escape(c)}</td><td class="num">{n}</td></tr>'
                        for c, n in sorted(by_code.items(), key=lambda kv: (-kv[1], kv[0])))
    flagged = [i for i in mine if i.code == "Review Clip" or i.labels.get("Note")]
    demo_note = ('<p style="background:#FFF4CC;padding:8px 10px;border-radius:6px;font-size:13px">'
                 'Demonstration pack built from a synthetic coding file to show the pipeline. '
                 'It is not an assessment of any real player.</p>') if demo else ""

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{html.escape(player)} · Review pack</title>
{FONTS}<style>{CSS}</style></head><body><div class="page">
<header><h1>{html.escape(player)}</h1><p>Development review · {html.escape(match)}</p></header><div class="bar"></div>
{demo_note}
<div class="tiles"><div><b>{len(mine)}</b><span>coded involvements</span></div>
<div><b>{eff}</b><span>effective ({good} of {good + bad})</span></div>
<div><b>{pressure}</b><span>pressure acts + tackles</span></div>
<div><b>{contest}</b><span>contests won ({cw} of {cw + cl})</span></div></div>
<h2>Development themes</h2>{"".join(theme_html) or "<p>No themes set for this player yet.</p>"}
<h2>Involvements by quarter</h2>{quarter_svg(mine)}
<h2>Coach-flagged moments</h2>{clip_list(flagged, vid) if flagged else "<p>None flagged.</p>"}
<h2>Every code</h2><table><thead><tr><th>Code</th><th style="text-align:right">Count</th></tr></thead><tbody>{code_rows}</tbody></table>
<h2>Review notes</h2><div class="notes">Agreed next focus:</div>
<footer>Generated automatically from the coded game by src/review_pack.py · Clip links open the match vision at the moment of each event.</footer>
</div></body></html>"""
    return page, log_rows


def slugify(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_") or "x"


def run(codes: Path, themes_path: Path, out_dir: Path, match: str | None = None, video: str | None = None,
        demo: bool = False) -> Path:
    info, inst = read_xml(codes)
    match = match or info.get("match") or codes.stem
    vid = video_id(video or info.get("video", ""))
    themes = {k: v for k, v in json.loads(themes_path.read_text()).items() if not k.startswith("_")}

    per_player: dict[str, list[Instance]] = defaultdict(list)
    for i in inst:
        per_player[i.player].append(i)

    pack_dir = out_dir / "packs" / slugify(match)
    pack_dir.mkdir(parents=True, exist_ok=True)
    all_log, links = [], []
    for player in sorted(per_player):
        page, log = render_pack(player, per_player[player], themes.get(player, {}).get("themes", []),
                                match, vid, demo)
        f = pack_dir / f"{slugify(player)}.html"
        f.write_text(page, encoding="utf-8")
        all_log += log
        links.append((player, f.name, len(per_player[player])))

    items = "".join(f'<li><a href="{f}">{html.escape(p)}</a> · {n} involvements</li>' for p, f, n in links)
    (pack_dir / "index.html").write_text(
        f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
        f'content="width=device-width, initial-scale=1"><title>Review packs · {html.escape(match)}</title>{FONTS}'
        f'<style>{CSS}</style></head><body><div class="page"><header><h1>Review packs</h1><p>{html.escape(match)}'
        f'</p></header><div class="bar"></div><ul style="font-size:17px;line-height:2">{items}</ul></div></body></html>',
        encoding="utf-8")

    log_path = out_dir / "development_log.csv"
    existing = []
    if log_path.exists():
        with log_path.open() as fh:
            existing = [r for r in csv.DictReader(fh) if r["match"] != match]  # re-running a game replaces it
    fields = ["match", "player", "theme", "kind", "value", "target", "instances", "met"]
    with log_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(existing + all_log)
    return pack_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--codes", type=Path, required=True)
    ap.add_argument("--themes", type=Path, default=Path("config/themes_example.json"))
    ap.add_argument("--out", type=Path, default=Path("outputs"))
    ap.add_argument("--match")
    ap.add_argument("--video")
    ap.add_argument("--demo", action="store_true", help="label packs as a demonstration")
    a = ap.parse_args()
    print("Packs written to", run(a.codes, a.themes, a.out, a.match, a.video, a.demo))
