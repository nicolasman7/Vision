import pandas as pd

from pipeline.prediction.dataset import DEF_GROUPS, OFF_GROUPS, _personnel_counts, build_dataset, parse_personnel


def test_parse_personnel():
    assert parse_personnel("1 RB, 1 TE, 3 WR") == {"RB": 1, "TE": 1, "WR": 3}
    # odd case: extra OL, a defender on offense, missing space after comma
    assert parse_personnel("6 OL, 1 RB, 1 TE, 2 WR,1 DB") == {"OL": 6, "RB": 1, "TE": 1, "WR": 2, "DB": 1}
    assert parse_personnel(None) == {}


def _play(play_id, **kw):
    row = dict(season=2022, game_id="G", play_id=play_id, week=1, posteam="A", defteam="B",
               down=1, ydstogo=10, yardline_100=75, qtr=1, quarter_seconds_remaining=900,
               half_seconds_remaining=1800, game_seconds_remaining=3600, score_differential=0,
               posteam_timeouts_remaining=3, defteam_timeouts_remaining=3, shotgun=1, no_huddle=0,
               goal_to_go=0, play_type="pass", qb_dropback=1, qb_kneel=0, qb_spike=0,
               two_point_attempt=0, aborted_play=0, run_location=None, pass_location="left",
               air_yards=8.0, run_gap=None, pass_length="short", yards_gained=12, epa=1.1, wpa=0.02)
    row["pass"], row["rush"] = 1, 0
    row.update(kw)
    return row


def test_build_dataset_filters_and_no_leaks():
    pbp = pd.DataFrame([
        _play(1),
        _play(2, play_type="run", qb_dropback=0, run_location="right", pass_location=None, air_yards=None,
              **{"pass": 0, "rush": 1}),
        _play(3, play_type="qb_kneel", qb_dropback=0, qb_kneel=1, **{"pass": 0, "rush": 0}),
        _play(4, play_type="no_play"),
    ])
    part = pd.DataFrame({
        "nflverse_game_id": "G", "play_id": [1, 2, 3, 4],
        "offense_formation": "SHOTGUN", "offense_personnel": "6 OL, 1 RB, 2 TE, 1 WR",
        "defense_personnel": "4 DL, 2 LB, 5 DB", "defenders_in_box": 7.0,
        "n_offense": 11, "n_defense": 11,
        "number_of_pass_rushers": 4.0, "time_to_throw": 2.5, "was_pressure": False, "route": "GO",
        "ngs_air_yards": 8.0, "defense_man_zone_type": "MAN", "defense_coverage_type": "COVER_1",
    })

    df = build_dataset(pbp, part)

    assert df["play_id"].tolist() == [1, 2]
    assert df["is_pass"].tolist() == [1, 0]
    assert df["direction"].tolist() == ["left", "right"]
    assert df.loc[0, ["off_ol", "off_rb", "off_te", "off_wr"]].tolist() == [6, 1, 2, 1]
    forbidden = {"number_of_pass_rushers", "time_to_throw", "was_pressure", "route", "ngs_air_yards",
                 "defense_man_zone_type", "defense_coverage_type", "yards_gained", "epa", "wpa"}
    assert not forbidden & set(df.columns)


def test_personnel_both_eras():
    off = _personnel_counts(pd.Series(["1 RB, 1 TE, 3 WR", "6 OL, 1 RB, 2 TE, 1 WR",
                                       "1 C, 2 G, 1 QB, 1 FB, 1 RB, 2 T, 1 TE, 2 WR"]), OFF_GROUPS, "off")
    assert off.values.tolist() == [[5, 1, 1, 3], [6, 1, 2, 1], [5, 2, 1, 2]]  # ol, rb, te, wr
    dfn = _personnel_counts(pd.Series(["4 DL, 2 LB, 5 DB",
                                       "3 CB, 2 DE, 2 DT, 1 FS, 1 MLB, 1 OLB, 1 SS"]), DEF_GROUPS, "def")
    assert dfn.values.tolist() == [[4, 2, 5], [4, 2, 5]]  # dl, lb, db


BANNED_FTN = ["is_play_action", "is_screen_pass", "is_rpo", "is_trick_play", "is_qb_out_of_pocket",
              "is_interception_worthy", "is_throw_away", "read_thrown", "is_catchable_ball",
              "is_contested_ball", "is_created_reception", "is_drop", "is_qb_sneak", "n_blitzers",
              "n_pass_rushers", "is_qb_fault_sack"]


def test_ftn_banned_columns_never_reach_output():
    pbp = pd.DataFrame([_play(1, season=2023)])
    part = pd.DataFrame({"nflverse_game_id": ["G"], "play_id": [1], "offense_formation": ["UNDER CENTER"],
                         "offense_personnel": ["1 C, 2 G, 1 QB, 1 RB, 2 T, 1 TE, 3 WR"],
                         "defense_personnel": ["3 CB, 2 DE, 2 DT, 1 FS, 1 MLB, 1 OLB, 1 SS"],
                         "defenders_in_box": [6], "n_offense": [11], "n_defense": [11]})
    ftn = pd.DataFrame({"nflverse_game_id": ["G"], "nflverse_play_id": [1], "starting_hash": ["L"],
                        "qb_location": [" U"], "n_offense_backfield": [0.0], "n_defense_box": [7],
                        "is_motion": [True], "is_no_huddle": [False], **{c: [1] for c in BANNED_FTN}})

    row = build_dataset(pbp, part, ftn).iloc[0]

    assert not (set(BANNED_FTN) | {f"ftn_{c}" for c in BANNED_FTN} | {"ftn_is_no_huddle"}) & set(row.index)
    assert row["ftn_qb_location"] == "U" and row["ftn_box"] == 7 and row["ftn_is_motion"] == 1
    assert row["formation_family"] == "under_center" and row["is_empty"] == 1
    assert pd.isna(row["offense_formation"]) and pd.isna(row["defenders_in_box"]) and row["box_part23"] == 6
