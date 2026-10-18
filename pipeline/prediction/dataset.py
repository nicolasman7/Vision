"""Build the play-predictor training table: one row per offensive snap, pre-snap features + labels."""

import re

import numpy as np
import pandas as pd

# Things you can know BEFORE the snap. Anything about what happened after (yards, EPA,
# pass rushers, coverage, routes) would leak the answer, so it never enters the table.
ID_COLS = ["game_id", "play_id", "week", "posteam", "defteam"]
PBP_FEATURES = [
    "down", "ydstogo", "yardline_100", "qtr",
    "quarter_seconds_remaining", "half_seconds_remaining", "game_seconds_remaining",
    "score_differential", "posteam_timeouts_remaining", "defteam_timeouts_remaining",
    "shotgun", "no_huddle", "goal_to_go",
]
PART_FEATURES = ["offense_formation", "offense_personnel", "defense_personnel",
                 "defenders_in_box", "n_offense", "n_defense"]
LABELS = ["is_pass", "direction", "air_yards"]


def parse_personnel(s):
    """'1 RB, 1 TE, 3 WR' -> {'RB': 1, 'TE': 1, 'WR': 3}.

    Real strings are messy: '6 OL, 1 RB, 2 TE, 1 WR' (extra lineman),
    '1 RB, 1 TE, 2 WR,1 DB' (a defender lined up on offense, no space after the comma),
    '2 QB, ...'. A regex for "<number> <letters>" handles all of them. None/NaN -> {}.
    """
    if not isinstance(s, str):
        return {}
    return {pos: int(n) for n, pos in re.findall(r"(\d+)\s*([A-Z]+)", s)}


def _personnel_counts(series, positions, prefix, defaults=None):
    """Turn a column of personnel strings into one integer column per position."""
    parsed = series.map(parse_personnel)
    out = {}
    for pos in positions:
        # The string only lists unusual groups: 5 OL is normal, so "6 OL" appears but "5 OL" never does.
        default = (defaults or {}).get(pos, 0)
        out[f"{prefix}_{pos.lower()}"] = [d.get(pos, default) if d else pd.NA for d in parsed]
    return pd.DataFrame(out, index=series.index).astype("Int64")


def filter_plays(pbp, verbose=False):
    """Keep only real offensive snaps: designed runs and dropbacks that counted."""
    steps = [
        # pass/rush flags are 1 on every run or dropback, including sacks and scrambles.
        ("pass or rush", lambda d: (d["pass"] == 1) | (d["rush"] == 1)),
        # no_play = a penalty wiped the play out; nflverse doesn't set qb_dropback on these, so no label.
        ("not no_play (penalty-nullified)", lambda d: d["play_type"] != "no_play"),
        # Kneels and spikes are clock management, not a real run/pass decision.
        ("not qb_kneel", lambda d: d["qb_kneel"] != 1),
        ("not qb_spike", lambda d: d["qb_spike"] != 1),
        # 2-pt tries start at the 2 and have no down/distance, a different situation.
        ("not 2-pt attempt", lambda d: d["two_point_attempt"] != 1),
        # Aborted = botched snap; we can't tell what was planned.
        ("not aborted", lambda d: d["aborted_play"] != 1),
        # Drops any remaining special-teams rows (kickoff/punt/FG/XP), keeps fakes labelled run/pass.
        ("play_type is run/pass", lambda d: d["play_type"].isin(["run", "pass"])),
    ]
    if verbose:
        print(f"{'start':<35}{len(pbp):>7}")
    for name, keep in steps:
        pbp = pbp[keep(pbp)]
        if verbose:
            print(f"{name:<35}{len(pbp):>7}")
    return pbp


def build_dataset(pbp, participation, verbose=False):
    plays = filter_plays(pbp, verbose)

    part = participation.rename(columns={"nflverse_game_id": "game_id"})[["game_id", "play_id"] + PART_FEATURES]
    df = plays.merge(part, on=["game_id", "play_id"], how="left", validate="one_to_one", indicator=True)
    if verbose:
        matched = (df["_merge"] == "both").mean()
        has_formation = df["offense_formation"].notna().mean()
        print(f"join rate: {matched:.2%} matched, {has_formation:.2%} have offense_formation")

    # Label = intent. qb_dropback is 1 whenever the QB dropped back to pass, even if he got
    # sacked or scrambled. play_type would call a scramble a "run", but the play call was a pass.
    df["is_pass"] = df["qb_dropback"].astype(int)
    # Direction of the designed play: pass_location for dropbacks, run_location for runs.
    # Sacks and scrambles have no pass_location, so they get NaN (scramble direction isn't the plan).
    df["direction"] = np.where(df["is_pass"] == 1, df["pass_location"], df["run_location"])
    df["direction"] = df["direction"].astype("string")

    # 0 players means NGS didn't record the play, not that nobody was on the field.
    for c in ["n_offense", "n_defense"]:
        df[c] = df[c].where(df[c] > 0).astype("Int64")

    off = _personnel_counts(df["offense_personnel"], ["OL", "RB", "TE", "WR"], "off", {"OL": 5})
    dfn = _personnel_counts(df["defense_personnel"], ["DL", "LB", "DB"], "def")

    return pd.concat([df[ID_COLS + PBP_FEATURES + PART_FEATURES], off, dfn, df[LABELS]], axis=1).reset_index(drop=True)
