import numpy as np
import pandas as pd

from scripts import analyze_buckets as ab


def _plays():
    # Team A: week 1 all left, week 2 all right, week 3 one of each. Team B in week 1 only (must not leak into A).
    return pd.DataFrame({
        "season": [2022] * 7, "posteam": ["A"] * 6 + ["B"], "week": [1, 1, 2, 2, 3, 3, 1],
        "is_pass": [0] * 7,
        "direction": pd.array(["left", "left", "right", "right", "left", "right", "middle"], dtype="string"),
    })


def test_tendency_uses_only_earlier_weeks():
    t = ab.team_tendency(_plays())
    assert t["team_run_left"].iloc[[0, 1, 6]].isna().all()  # week 1: no history yet
    assert (t["team_run_left"].iloc[2:4] == 1.0).all()  # week 2 sees only week 1 (all left)
    assert (t["team_run_left"].iloc[4:6] == 0.5).all()  # week 3 sees weeks 1-2
    assert t["team_pass_left"].isna().all()  # no passes yet -> no pass tendency


def test_tendency_ignores_current_and_future_weeks():
    df = _plays()
    before = ab.team_tendency(df)
    changed = df.copy()
    changed.loc[changed["week"] >= 3, "direction"] = "middle"  # rewrite week 3's outcomes
    after = ab.team_tendency(changed)
    pd.testing.assert_frame_equal(before, after)  # week 3's own features can't see week 3


def test_field_relative_mapping():
    d = pd.Series(["left", "right", "middle", "left", "left", None], dtype=object)
    h = pd.Series(["L", "L", "R", "R", "M", "L"], dtype="string")
    out = ab.field_relative(d, h)
    assert out.iloc[:4].tolist() == ["hash_side", "wide_side", "middle", "wide_side"]
    assert out.iloc[4:].isna().all()  # middle hash and unknown direction have no field-relative label
    assert np.all(out.dropna().isin(["hash_side", "wide_side", "middle"]))
