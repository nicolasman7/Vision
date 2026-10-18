"""Build the play-predictor training table: one row per offensive snap, pre-snap features + labels."""

import re

import numpy as np
import pandas as pd

# Things you can know BEFORE the snap. Anything about what happened after (yards, EPA,
# pass rushers, coverage, routes) would leak the answer, so it never enters the table.
ID_COLS = ["season", "game_id", "play_id", "week", "posteam", "defteam"]
PBP_FEATURES = [
    "down", "ydstogo", "yardline_100", "qtr",
    "quarter_seconds_remaining", "half_seconds_remaining", "game_seconds_remaining",
    "score_differential", "posteam_timeouts_remaining", "defteam_timeouts_remaining",
    "shotgun", "no_huddle", "goal_to_go",
]
PART_FEATURES = ["offense_formation", "offense_personnel", "defense_personnel",
                 "defenders_in_box", "n_offense", "n_defense"]
# FTN charting (2022+), hand-charted pre-snap details. Everything else in FTN (play action, screen,
# RPO, blitzers, drops...) describes what happened after the snap, so it is never read.
FTN_FEATURES = ["starting_hash", "qb_location", "n_offense_backfield", "n_defense_box", "is_motion"]
# Labels. run_gap and pass_length describe the play itself: targets only, never features.
LABELS = ["is_pass", "direction", "air_yards", "run_gap", "pass_length"]
# Columns read from play-by-play (it has 372; reading only these keeps memory low).
PBP_COLUMNS = list(dict.fromkeys(
    ID_COLS + PBP_FEATURES
    + ["pass", "rush", "play_type", "qb_kneel", "qb_spike", "two_point_attempt", "aborted_play"]
    + ["qb_dropback", "pass_location", "run_location", "air_yards", "run_gap", "pass_length"]))

# Personnel groups. 2016-2022 (NGS) strings say "1 RB, 1 TE, 3 WR"; 2023+ list exact positions,
# e.g. "1 C, 2 G, 1 QB, 1 RB, 2 T, 1 TE, 3 WR" and "3 CB, 2 DE, 2 DT, 1 FS, 1 MLB, 1 OLB, 1 SS".
# Summing the detailed positions into the old groups makes both eras comparable.
# FB counts as RB because NGS strings never list FB separately.
OFF_GROUPS = {"ol": ["OL", "C", "G", "T"], "rb": ["RB", "FB"], "te": ["TE"], "wr": ["WR"]}
DEF_GROUPS = {"dl": ["DL", "DE", "DT", "NT"], "lb": ["LB", "ILB", "MLB", "OLB"],
              "db": ["DB", "CB", "S", "FS", "SS"]}
FTN_FORMATION = {"SHOTGUN": "shotgun", "UNDER CENTER": "under_center", "PISTOL": "pistol"}


def parse_personnel(s):
    """'1 RB, 1 TE, 3 WR' -> {'RB': 1, 'TE': 1, 'WR': 3}.

    Real strings are messy: '6 OL, 1 RB, 2 TE, 1 WR' (extra lineman),
    '1 RB, 1 TE, 2 WR,1 DB' (a defender lined up on offense, no space after the comma),
    '2 QB, ...'. A regex for "<number> <letters>" handles all of them. None/NaN -> {}.
    """
    if not isinstance(s, str):
        return {}
    return {pos: int(n) for n, pos in re.findall(r"(\d+)\s*([A-Z]+)", s)}


def _personnel_counts(series, groups, prefix):
    """Turn a column of personnel strings into one integer column per group (summing its positions)."""
    parsed = series.map(parse_personnel)
    out = {}
    for name, positions in groups.items():
        col = []
        for d in parsed:
            if not d:
                col.append(pd.NA)
            elif name == "ol" and not any(p in d for p in positions):
                # NGS strings only list unusual lines: 5 OL is normal, so "6 OL" appears but "5 OL" never does.
                col.append(5)
            else:
                col.append(sum(d.get(p, 0) for p in positions))
        out[f"{prefix}_{name}"] = col
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


def _box(s):
    # A box count of 0 on a real snap means "not recorded", not an empty box.
    return s.where(s > 0).astype("Int64")


def build_dataset(pbp, participation, ftn=None, verbose=False):
    """One season in, one table out. ftn is None for seasons before 2022 (FTN charting starts then)."""
    plays = filter_plays(pbp, verbose)

    part = participation.rename(columns={"nflverse_game_id": "game_id"})[["game_id", "play_id"] + PART_FEATURES]
    df = plays.merge(part, on=["game_id", "play_id"], how="left", validate="one_to_one", indicator=True)
    if verbose:
        matched = (df["_merge"] == "both").mean()
        has_formation = df["offense_formation"].notna().mean()
        print(f"join rate: {matched:.2%} matched, {has_formation:.2%} have offense_formation")

    # Only the whitelisted pre-snap FTN columns are selected; the rest of FTN never touches the table.
    ftn_names = {c: f"ftn_{c}" for c in FTN_FEATURES} | {"n_defense_box": "ftn_box"}
    if ftn is None:
        for c in ftn_names.values():
            df[c] = np.nan
    else:
        ftn = ftn.rename(columns={"nflverse_game_id": "game_id", "nflverse_play_id": "play_id"})
        ftn = ftn[["game_id", "play_id"] + FTN_FEATURES].rename(columns=ftn_names)
        df = df.drop(columns="_merge").merge(ftn, on=["game_id", "play_id"], how="left", validate="one_to_one",
                                             indicator=True)
        if verbose:
            print(f"FTN join rate: {(df['_merge'] == 'both').mean():.2%}")

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

    # 2023+ participation uses a different formation vocabulary (SHOTGUN / UNDER CENTER / PISTOL),
    # so the raw NGS column only exists through 2022; formation_family below covers every season.
    ngs = df["season"] <= 2022
    raw_formation = df["offense_formation"]
    df["offense_formation"] = raw_formation.where(ngs)
    df["formation_family"] = np.where(
        ngs,
        np.where(raw_formation == "PISTOL", "pistol", np.where(df["shotgun"] == 1, "shotgun", "under_center")),
        raw_formation.map(FTN_FORMATION))
    df["formation_family"] = df["formation_family"].astype("string")  # unmapped/missing -> <NA>
    # Empty backfield: NGS calls it a formation; FTN counts backfield players. NaN when unknown.
    ngs_empty = (raw_formation == "EMPTY").where(raw_formation.notna())
    ftn_empty = (df["ftn_n_offense_backfield"] == 0).where(df["ftn_n_offense_backfield"].notna())
    df["is_empty"] = ngs_empty.where(ngs, ftn_empty).astype("float64")

    # Box counts are kept per source, never merged: NGS (<=2022), 2023+ participation and FTN each
    # count "the box" their own way, and mixing them would make one column mean different things.
    df["box_part23"] = _box(df["defenders_in_box"].where(~ngs))
    df["defenders_in_box"] = _box(df["defenders_in_box"].where(ngs))
    df["ftn_box"] = _box(df["ftn_box"])
    # FTN uses "0" for "not charted" and has stray spaces (" S").
    for c in ["ftn_starting_hash", "ftn_qb_location"]:
        df[c] = df[c].astype("string").str.strip().replace("0", pd.NA)
    df["ftn_n_offense_backfield"] = df["ftn_n_offense_backfield"].astype("Int64")
    df["ftn_is_motion"] = df["ftn_is_motion"].astype("boolean").astype("Int64")

    off = _personnel_counts(df["offense_personnel"], OFF_GROUPS, "off")
    dfn = _personnel_counts(df["defense_personnel"], DEF_GROUPS, "def")

    cols = ID_COLS + PBP_FEATURES + PART_FEATURES + ["formation_family", "is_empty", "box_part23"] \
        + list(ftn_names.values())
    return pd.concat([df[cols], off, dfn, df[LABELS]], axis=1).reset_index(drop=True)
