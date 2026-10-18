"""Multi-season run/pass experiments (E1-E5). Run from repo root: uv run python -m scripts.experiments_prediction

E1 more seasons, E2 newest season, E3 FTN features, E4 situation ablation, E5 robustness to vision errors.
Writes models/metrics_multiseason.json and models/run_pass_multiseason.joblib.
"""

import json

import joblib
import numpy as np
import pandas as pd

from pipeline.prediction import model as m

df = pd.read_parquet("data/processed/plays_2016_2025.parquet")
results = {}


def compare(train, test, feature_sets):
    """Fit one model per feature set on the same train/test. None = always-pass baseline."""
    y, rows, models = test["is_pass"], {}, {}
    for name, cols in feature_sets.items():
        if cols is None:
            p = np.full(len(test), train["is_pass"].mean())
        else:
            models[name] = m.train_run_pass(train, cols)
            p = models[name].predict_proba(m.features(test, cols))[:, 1]
        rows[name] = {**m.evaluate(y, p), "train_plays": len(train), "test_plays": len(test)}
        if cols is not None:
            rows[name]["_p"] = p
    return rows, models


def show(title, rows):
    print(f"\n{title}")
    table = pd.DataFrame({k: {c: v for c, v in r.items() if c != "_p"} for k, r in rows.items()}).T
    print(table.round(4).to_string())


def calib(y, p, title):
    table = m.calibration_table(y, p)
    print(f"\ncalibration, {title}")
    print(pd.DataFrame(table).round(3).to_string(index=False))
    return table


def strip(rows):
    return {k: {c: v for c, v in r.items() if c != "_p"} for k, r in rows.items()}


# --- E1: does adding older seasons help? Same test as the 2022 model: 2022 weeks 15-22. ---
e1, e1_models, e1_train = {}, {}, {}
for name, seasons in {"a_2022_only": [2022],
                      "b_2016_2022": list(range(2016, 2023)),
                      "c_2019_2022": list(range(2019, 2023))}.items():
    train, test_e1 = m.season_split(df, 2022, last_train_week=14, train_seasons=seasons)
    rows, models = compare(train, test_e1, {name: m.ALL_FEATURES})
    e1 |= rows
    e1_models[name], e1_train[name] = models[name], train
e1 = {"always_pass": compare(e1_train["a_2022_only"], test_e1, {"x": None})[0]["x"]} | e1
show("E1  more seasons (test: 2022 weeks 15-22, features: ALL_FEATURES)", e1)
best = min(e1_models, key=lambda k: e1[k]["log_loss"])
print(f"best by log loss: {best}")
results["E1_more_seasons"] = strip(e1) | {"best": best,
                                          "calibration_best": calib(test_e1["is_pass"], e1[best]["_p"], f"E1 {best}")}

# --- E2: newest season. Train 2016-2024, test all of 2025, era-robust features only. ---
train_e2, test_e2 = m.season_split(df, 2025)
e2, e2_models = compare(train_e2, test_e2, {
    "always_pass": None, "situation_only": m.SITUATION, "era_features": m.ERA_FEATURES})
show("E2  newest season (train 2016-2024, test 2025)", e2)
results["E2_newest_season"] = strip(e2) | {
    "calibration_era_features": calib(test_e2["is_pass"], e2["era_features"]["_p"], "E2 era_features")}

# --- E3: FTN hand-charted pre-snap columns (2022+ only), train 2022-2024, test 2025. ---
train_e3, test_e3 = m.season_split(df, 2025, train_seasons=[2022, 2023, 2024])
base = m.SITUATION + m.ERA_ALIGNMENT
e3, _ = compare(train_e3, test_e3, {
    "without_ftn": base, "with_ftn": base + m.FTN, "with_ftn_and_motion (borderline)": base + m.FTN + m.FTN_MOTION})
show("E3  FTN features (train 2022-2024, test 2025)", e3)
results["E3_ftn"] = strip(e3)

# --- E4: how much does each piece of the situation add? (Evidence for scorebug vs. user input.) ---
align = m.ERA_ALIGNMENT + m.FTN
clocks = ["qtr", "quarter_seconds_remaining", "half_seconds_remaining", "game_seconds_remaining"]
steps = {"alignment_only": align}
steps["+ down, ydstogo"] = steps["alignment_only"] + ["down", "ydstogo"]
steps["+ yardline_100, goal_to_go"] = steps["+ down, ydstogo"] + ["yardline_100", "goal_to_go"]
steps["+ score_differential, qtr, clocks"] = steps["+ yardline_100, goal_to_go"] + ["score_differential"] + clocks
steps["full (+ timeouts)"] = steps["+ score_differential, qtr, clocks"] + [
    "posteam_timeouts_remaining", "defteam_timeouts_remaining"]
assert set(steps["full (+ timeouts)"]) == set(m.ERA_FEATURES)
e4, _ = compare(train_e2, test_e2, steps)
show("E4  situation ablation (train 2016-2024, test 2025, cumulative)", e4)
results["E4_situation_ablation"] = strip(e4)

# --- E5: robustness to vision errors, best E1 model on the E1 test set. ---
model, X = e1_models[best], m.features(test_e1, m.ALL_FEATURES)
train_formations = e1_train[best]["offense_formation"]
y = test_e1["is_pass"]


def score(Xc):
    return m.evaluate(y, model.predict_proba(Xc)[:, 1])


e5 = {"formation_misread": {}, "box_off_by_one": {}}
for frac in [0.0, 0.1, 0.2, 0.3]:
    e5["formation_misread"][f"{frac:.0%}"] = score(m.corrupt_category(X, "offense_formation", frac, train_formations))
    e5["box_off_by_one"][f"{frac:.0%}"] = score(m.shift_count(X, "defenders_in_box", frac))
no_formation = X.copy()
no_formation["offense_formation"] = pd.Categorical([np.nan] * len(X), categories=m.FORMATIONS)
e5["formation_unreadable_all_nan"] = score(no_formation)
print(f"\nE5  robustness, model {best} on 2022 weeks 15-22")
for k in ["formation_misread", "box_off_by_one"]:
    print(f"{k}:\n{pd.DataFrame(e5[k]).T.round(4).to_string()}")
print(f"formation unreadable (all NaN): {pd.Series(e5['formation_unreadable_all_nan']).round(4).to_dict()}")
results["E5_robustness"] = e5

# Saved model: the E2 era-robust one. Its features exist in every season (offense_formation stops in 2022)
# and it was trained on the most recent data, so it is the one to carry forward.
joblib.dump(e2_models["era_features"], "models/run_pass_multiseason.joblib")
results["saved_model"] = {"path": "models/run_pass_multiseason.joblib", "experiment": "E2 era_features",
                          "train": "2016-2024", "features": m.ERA_FEATURES}
with open("models/metrics_multiseason.json", "w") as f:
    json.dump(results, f, indent=2, default=float)
print("\nsaved models/run_pass_multiseason.joblib, models/metrics_multiseason.json")
