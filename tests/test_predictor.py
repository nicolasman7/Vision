import numpy as np
import pandas as pd
import pytest

from pipeline.prediction import model as m
from pipeline.prediction import predictor


@pytest.fixture(scope="module")
def bundle():
    """A tiny bundle trained on random plays: enough to exercise the real train -> predict path."""
    rng = np.random.default_rng(0)
    n = 600
    df = pd.DataFrame({c: rng.integers(0, 5, n).astype(float) for c in m.V3_INPUTS})
    df["formation_family"] = rng.choice(["shotgun", "under_center", "pistol"], n)
    df["ftn_starting_hash"] = rng.choice(["L", "M", "R"], n)
    df["is_pass"] = (df["formation_family"] == "shotgun").astype(int)
    df["direction"] = rng.choice(["left", "middle", "right"], n)
    df["air_yards"] = np.where(df["is_pass"] == 1, rng.integers(-5, 30, n), np.nan)
    return m.train_v3(df.iloc[:400], df.iloc[400:])


def test_predict_with_no_inputs(bundle):
    out = predictor.predict(bundle)
    assert 0 < out["p_pass"] < 1
    for head in ("direction", "depth"):
        assert list(out[head]) == bundle["heads"][head]["classes"]
        assert sum(out[head].values()) == pytest.approx(1)


def test_omitted_equals_none(bundle):
    given = dict(down=3, ydstogo=7, is_empty=0)
    assert predictor.predict(bundle, **given) == predictor.predict(bundle, **given, formation_family=None)


def test_onehot_missing_is_all_nan():
    X = m.onehot(pd.DataFrame({"formation_family": ["pistol", None, "bogus"]}), ["formation_family"])
    assert X.iloc[0].tolist() == [0.0, 0.0, 1.0]
    assert X.iloc[1:].isna().all().all()  # missing and unknown values -> NaN in every one-hot column


def test_unknown_input_rejected(bundle):
    with pytest.raises(ValueError):
        predictor.predict(bundle, dwn=3)  # typo
