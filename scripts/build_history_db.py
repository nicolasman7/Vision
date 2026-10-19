"""Build coreml/history.sqlite: every play 2016-2025 for the app's "Play lookup" screen, plus a parity fixture.

Run from repo root: uv run python -m scripts.build_history_db
Needs data/processed/plays_<year>.parquet (scripts.build_dataset) and models/predictor_v3.joblib.
"""

import json
import os
import sqlite3

import numpy as np
import pandas as pd

from pipeline.prediction.predictor import load, predict_frame

SEASONS = range(2016, 2026)
DB = "coreml/history.sqlite"
FIXTURE = "app/VisionTests/Fixtures/history_parity.json"
LAST_TRAIN_SEASON = 2023  # v3 trained on 2016-2023, calibrated on 2024, tested on 2025
DESC_FROM = 2022  # play text only from this season on, to keep the app bundle small

SCHEMA = """
CREATE TABLE games(game_id TEXT PRIMARY KEY, season INT, week INT, home TEXT, away TEXT, game_date TEXT);
CREATE TABLE plays(game_id TEXT, play_id INT, qtr INT, clock_s INT, down INT, ydstogo INT, yardline_100 INT,
  posteam TEXT, defteam TEXT, score_differential INT, posteam_timeouts INT, defteam_timeouts INT, no_huddle INT,
  formation_family TEXT, is_empty INT, backfield INT, off_ol INT, off_rb INT, off_te INT, off_wr INT,
  def_dl INT, def_lb INT, def_db INT, box INT, hash TEXT, is_pass INT, direction TEXT, air_yards REAL,
  depth_bucket TEXT, desc TEXT, in_training INT);
CREATE INDEX plays_lookup ON plays(game_id, qtr, clock_s);
"""
# DB column -> parquet column, for the ones that are just renamed.
RENAME = {"clock_s": "quarter_seconds_remaining", "posteam_timeouts": "posteam_timeouts_remaining",
          "defteam_timeouts": "defteam_timeouts_remaining", "backfield": "ftn_n_offense_backfield",
          "box": "ftn_box", "hash": "ftn_starting_hash"}
PLAY_COLS = ["game_id", "play_id", "qtr", "clock_s", "down", "ydstogo", "yardline_100", "posteam", "defteam",
             "score_differential", "posteam_timeouts", "defteam_timeouts", "no_huddle", "formation_family",
             "is_empty", "backfield", "off_ol", "off_rb", "off_te", "off_wr", "def_dl", "def_lb", "def_db", "box",
             "hash", "is_pass", "direction", "air_yards", "depth_bucket", "desc", "in_training"]
TEXT_COLS = {"game_id", "posteam", "defteam", "formation_family", "hash", "direction", "depth_bucket", "desc"}


def season_rows(year):
    plays = pd.read_parquet(f"data/processed/plays_{year}.parquet")
    pbp = pd.read_parquet(f"data/raw/nflverse/play_by_play_{year}.parquet",
                          columns=["game_id", "play_id", "desc", "home_team", "away_team", "game_date"])
    df = plays.merge(pbp, on=["game_id", "play_id"], how="left", validate="one_to_one")
    df = df.rename(columns={v: k for k, v in RENAME.items()})
    # Text is ~half the file; with every season it came to ~75 MB, so only 2022+ keep the play text.
    df["desc"] = df["desc"].str.slice(0, 160) if year >= DESC_FROM else None
    # Same buckets as the v3 depth head: only throws have one (sacks/scrambles have no air_yards).
    ay = df["air_yards"].where(df["is_pass"] == 1)
    df["depth_bucket"] = pd.Series(np.select([ay <= 0, ay < 10], ["<=0", "1-9"], "10+"), index=df.index).where(ay.notna())
    df["in_training"] = int(year <= LAST_TRAIN_SEASON)
    for c in set(PLAY_COLS) - TEXT_COLS - {"air_yards"}:
        df[c] = df[c].astype("float64").round().astype("Int64")  # whole numbers in, NULL stays NULL
    games = df.groupby("game_id", sort=False).first().reset_index()
    games = games[["game_id", "season", "week", "home_team", "away_team", "game_date"]]
    return games, df[PLAY_COLS]


def records(df):
    # pandas NA/NaN -> None, so sqlite writes NULL (unknown), never 0.
    return df.astype(object).where(df.notna(), None).itertuples(index=False, name=None)


def db_inputs(rows):
    """DB rows -> predictor inputs, exactly as the Swift side reads them (derived inputs left for derive())."""
    inputs = rows.rename(columns=RENAME).drop(columns=["game_id", "play_id"])
    return inputs.astype({c: "float64" for c in inputs.columns if c not in ("formation_family", "ftn_starting_hash")})


def write_fixture(con):
    cols = ["game_id", "play_id", "qtr", "clock_s", "down", "ydstogo", "yardline_100", "score_differential",
            "posteam_timeouts", "defteam_timeouts", "no_huddle", "formation_family", "is_empty", "backfield",
            "off_ol", "off_rb", "off_te", "off_wr", "def_dl", "def_lb", "def_db", "box", "hash"]
    # One mid-season play from 8 seasons: 5 without FTN columns (pre-2022), 3 with.
    rows = pd.concat([pd.read_sql(f"SELECT {', '.join(cols)} FROM plays WHERE substr(game_id, 1, 4) = '{y}' "
                                  "LIMIT 1 OFFSET 17000", con)
                      for y in (2016, 2017, 2019, 2020, 2021, 2022, 2024, 2025)], ignore_index=True)
    pred = predict_frame(load(), db_inputs(rows))
    out = [{"game_id": r.game_id, "play_id": int(r.play_id),
            "p_pass": p["p_pass"],
            "direction": {c: p[f"direction_{c}"] for c in ("left", "middle", "right")},
            "depth": {c: p[f"depth_{c}"] for c in ("<=0", "1-9", "10+")}}
           for r, (_, p) in zip(rows.itertuples(), pred.iterrows())]
    with open(FIXTURE, "w") as f:
        json.dump(out, f, indent=1)


def main():
    if os.path.exists(DB):
        os.remove(DB)
    con = sqlite3.connect(DB)
    con.executescript(SCHEMA)
    for year in SEASONS:
        games, plays = season_rows(year)
        con.executemany("INSERT INTO games VALUES (?,?,?,?,?,?)", records(games))
        con.executemany(f"INSERT INTO plays VALUES ({','.join('?' * len(PLAY_COLS))})", records(plays))
        print(f"{year}: {len(games)} games, {len(plays)} plays")
    con.commit()
    write_fixture(con)
    con.execute("VACUUM")
    con.close()
    print(f"wrote {DB}: {os.path.getsize(DB) / 1e6:.1f} MB; {FIXTURE}")


if __name__ == "__main__":
    main()
