"""End-to-end checks on a simulated dataset (never on reported results) plus review-pack checks.
Run: python -m pytest -q
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def run(mod, cwd):
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    subprocess.run([sys.executable, "-W", "ignore", "-m", mod], cwd=cwd, env=env, check=True, capture_output=True)


def test_full_pipeline_on_simulated_data(tmp_path):
    subprocess.run([sys.executable, "-m", "tests.make_fake_raw", str(tmp_path)], cwd=ROOT, check=True, capture_output=True)
    for m in ["src.load", "src.translate", "src.readiness", "src.figures"]:
        run(m, tmp_path)
    g = pd.read_parquet(tmp_path / "data/processed/player_games.parquet")
    assert not g.duplicated(["match_id", "player_id"]).any()          # one row per player per game
    assert set(g.comp) == {"AFL", "VFL"}
    ft = pd.read_csv(tmp_path / "outputs/translation_factors.csv")
    assert ((ft.lo <= ft.factor) & (ft.factor <= ft.hi)).all()         # point estimate inside its interval
    # the simulation inflates VFL output by 1.3x, so factors must come out below 1
    assert (ft.factor < 1).all()
    v = pd.read_csv(tmp_path / "outputs/translation_validation.csv")
    assert v.n_test.min() > 0
    site = json.loads((tmp_path / "outputs/site_data.json").read_text())
    assert site["players"] and site["readiness"]["effects"]


def test_review_pack(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    subprocess.run([sys.executable, "-m", "src.review_pack", "--codes", str(ROOT / "examples/demo_codes.xml"),
                    "--themes", str(ROOT / "config/themes_example.json"), "--out", str(tmp_path),
                    "--video", "https://youtu.be/AAAAAAAAAAA", "--demo"], cwd=ROOT, env=env, check=True)
    packs = list((tmp_path / "packs").rglob("Player_*.html"))
    assert packs
    html = packs[0].read_text()
    assert "youtube.com/watch?v=AAAAAAAAAAA&t=" in html                 # clips deep-link to the vision
    log = pd.read_csv(tmp_path / "development_log.csv")
    assert {"match", "player", "theme", "value", "target", "met"} <= set(log.columns)
    # re-running the same game replaces its rows rather than duplicating them
    subprocess.run([sys.executable, "-m", "src.review_pack", "--codes", str(ROOT / "examples/demo_codes.xml"),
                    "--themes", str(ROOT / "config/themes_example.json"), "--out", str(tmp_path)], cwd=ROOT, env=env, check=True)
    assert len(pd.read_csv(tmp_path / "development_log.csv")) == len(log)
