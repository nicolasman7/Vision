"""Train and evaluate the pre-snap play predictor. Run from repo root: uv run python -m scripts.train_prediction"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from pipeline.prediction import model as m

df = pd.read_parquet("data/processed/plays_2022.parquet")
train, test = m.time_split(df)
print(f"train weeks 1-14: {len(train)} plays, {train.game_id.nunique()} games")
print(f"test weeks 15-22: {len(test)} plays, {test.game_id.nunique()} games")
y = test["is_pass"]
results = {"split": {"train_plays": len(train), "test_plays": len(test)}}

# --- Run/pass: A majority, B situation only, C situation + alignment ---
base_rate = train["is_pass"].mean()  # predict "pass" with the training pass rate as the probability
model_b = m.train_run_pass(train, m.SITUATION)
model_c = m.train_run_pass(train, m.ALL_FEATURES)
p = {
    "A_majority": np.full(len(test), base_rate),
    "B_situation": model_b.predict_proba(m.features(test, m.SITUATION))[:, 1],
    "C_situation_alignment": model_c.predict_proba(m.features(test, m.ALL_FEATURES))[:, 1],
}
results["run_pass"] = {k: m.evaluate(y, v) for k, v in p.items()}
print(f"\nRUN/PASS (test pass rate {y.mean():.1%})")
print(pd.DataFrame(results["run_pass"]).T.round(4).to_string())

# --- Calibration of C. Recalibration needs data the model didn't train on, so: train on weeks 1-10,
# fit the isotonic map on 11-14, and compare on the same test weeks 15-22. ---
results["calibration_C"] = m.calibration_table(y, p["C_situation_alignment"])
fit, calib = m.time_split(train, last_train_week=10)
model_c10 = m.train_run_pass(fit, m.ALL_FEATURES)
cal = m.calibrate(model_c10, calib, m.ALL_FEATURES)
X_test = m.features(test, m.ALL_FEATURES)
p10, pcal = model_c10.predict_proba(X_test)[:, 1], cal.predict_proba(X_test)[:, 1]
results["calibration_experiment"] = {
    "C_weeks1-10_raw": {**m.evaluate(y, p10), "table": m.calibration_table(y, p10)},
    "C_weeks1-10_isotonic_on_11-14": {**m.evaluate(y, pcal), "table": m.calibration_table(y, pcal)},
}
print("\nCALIBRATION, model C (trained weeks 1-14)")
print(pd.DataFrame(results["calibration_C"]).round(3).to_string(index=False))
for name, r in results["calibration_experiment"].items():
    print(f"\n{name}: acc {r['accuracy']:.4f}  log loss {r['log_loss']:.4f}  brier {r['brier']:.4f}")
    print(pd.DataFrame(r["table"]).round(3).to_string(index=False))

# --- Direction (left/middle/right), on plays where it's known ---
model_dir = m.train_direction(train)
known = test[test["direction"].notna()]
yd = known["direction"].astype(str).to_numpy()
pred = model_dir.predict(m.features(known, m.DIRECTION_FEATURES))
majority = train["direction"].mode()[0]
# Fairer baseline: the model knows run vs pass, so the baseline gets to know it too.
maj_by_type = train.groupby("is_pass")["direction"].agg(lambda s: s.mode()[0])
pred_by_type = known["is_pass"].map(maj_by_type).to_numpy()
is_pass = known["is_pass"].to_numpy() == 1
groups = {"overall": slice(None), "runs": ~is_pass, "passes": is_pass}
results["direction"] = {
    "test_rows": len(known),
    "test_distribution": known["direction"].value_counts(normalize=True).round(4).to_dict(),
    "model": {g: float((pred[s] == yd[s]).mean()) for g, s in groups.items()},
    f"baseline_always_{majority}": {g: float((yd[s] == majority).mean()) for g, s in groups.items()},
    "baseline_majority_by_run_pass": {g: float((pred_by_type[s] == yd[s]).mean()) for g, s in groups.items()},
}
print(f"\nDIRECTION ({len(known)} test plays with known direction), accuracy")
print(pd.DataFrame({k: v for k, v in results["direction"].items() if isinstance(v, dict) and "overall" in v})
      .T.round(4).to_string())

# --- What matters: shuffle one column at a time on the test set and see how much log loss worsens ---
imp = permutation_importance(model_c, X_test, y, scoring="neg_log_loss", n_repeats=5, random_state=0)
top = pd.Series(imp.importances_mean, index=m.ALL_FEATURES).sort_values(ascending=False)
results["importance_C_logloss_increase"] = top.round(5).to_dict()
print("\nTOP 10 FEATURES (log loss increase when shuffled)")
print(top.head(10).round(4).to_string())

joblib.dump(model_c, "models/run_pass_2022.joblib")
joblib.dump(model_dir, "models/direction_2022.joblib")
with open("models/metrics_2022.json", "w") as f:
    json.dump(results, f, indent=2, default=float)
print("\nsaved models/run_pass_2022.joblib, models/direction_2022.joblib, models/metrics_2022.json")
