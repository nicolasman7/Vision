"""Export the v3 predictor bundle to Core ML and prove the exported models match Python.
Run: uv run python -m scripts.export_coreml

Writes coreml/run_pass.mlmodel, direction.mlmodel and depth.mlmodel (calibration folded into the trees)
and coreml/predictor_schema.json, the input contract for the Swift app.
Then it runs 2,000 plays from 2025 (some with inputs blanked) through Core ML and through
predictor.predict_frame, and fails if any probability differs by more than 1e-4.
"""

import json
import os
import time

import coremltools as ct
import numpy as np
import pandas as pd

from pipeline.export.hgb_to_coreml import convert
from pipeline.prediction import model as m
from pipeline.prediction import predictor

OUT = "coreml"
TOL = 1e-4

bundle = predictor.load()
features = bundle["features"]
os.makedirs(OUT, exist_ok=True)

# --- Export ---
paths = {}
for name, head in bundle["heads"].items():
    c = head["calib"]  # run_pass: Platt {"a", "b"}; 3-class heads: temperature {"T"}. calib_fitted_2024 is unused.
    calib = (c["a"], c["b"]) if name == "run_pass" else c["T"]
    mlmodel = convert(head["model"], features, head["classes"], calib)
    mlmodel.short_description = f"Vision predictor v{bundle['version']}: {name} head"
    paths[name] = f"{OUT}/{name}.mlmodel"
    mlmodel.save(paths[name])

schema = {
    "version": bundle["version"],
    "input": "features: MLMultiArray of doubles, length len(features), in this exact order. "
             "Missing / unknown = NaN (the trees have a learned branch for missing; never use 0 for unknown).",
    "features": features,
    "onehot": {
        col: {"columns": [f"{col}={v}" for v in values],
              "rule": f"value v -> 1.0 in '{col}=v', 0.0 in the others; missing or not one of {values} -> NaN in all"}
        for col, values in bundle["onehot"].items()},
    "sources": bundle["sources"],
    # Same formulas as predictor.derive. Only fill a derived input when it wasn't given directly.
    "derived": {
        "goal_to_go": "1.0 if ydstogo == yardline_100 else 0.0; NaN if either is missing",
        "half_seconds_remaining": "quarter_seconds_remaining + (900 if qtr in {1, 3} else 0); NaN if qtr missing",
        "game_seconds_remaining": "quarter_seconds_remaining + 900 * max(0, 4 - qtr); NaN if either missing "
                                  "(overtime, qtr 5, counts only the quarter)",
    },
    "heads": {
        name: {"file": f"{name}.mlmodel", "classes": head["classes"],
               "outputs": {"label": "most likely class", "probs": "class -> probability (calibrated)"}}
        for name, head in bundle["heads"].items()},
    "conditioning": {
        "run_pass": "P(pass) = probs[1]",
        "direction": "conditional on the play being a run",
        "depth": "conditional on the ball being thrown",
    },
}
with open(f"{OUT}/predictor_schema.json", "w") as f:
    json.dump(schema, f, indent=2)

# --- Parity on 2025 ---
df = pd.read_parquet("data/processed/plays_2016_2025.parquet")
inputs = df[df.season == 2025].sample(2000, random_state=0)[list(bundle["sources"])].reset_index(drop=True)
situation = sorted({c for b in m.GROUPS["situation"] for c in m.BLOCKS[b]})
inputs.loc[0:199, m.BLOCKS["formation"]] = np.nan
inputs.loc[200:399, situation] = np.nan
inputs.loc[400:409, :] = np.nan  # nothing known at all

# C order: DataFrame.to_numpy gives a column-major array, and Core ML rejects a non-contiguous row
# ("value type not convertible").
X = np.ascontiguousarray(predictor.matrix(bundle, inputs).to_numpy("float64"))
ref = predictor.predict_frame(bundle, inputs)

print(f"{len(X)} plays from 2025 (200 no formation, 200 no situation, 10 fully blank)\n")
print(f"{'head':<10}{'max |diff|':>12}{'argmax agree':>14}{'size KB':>10}{'ms/row':>9}")
ok = True
for name, head in bundle["heads"].items():
    mlmodel = ct.models.MLModel(paths[name])
    start = time.perf_counter()
    out = [mlmodel.predict({"features": row}) for row in X]  # one call per row, like the app will do
    ms = (time.perf_counter() - start) / len(X) * 1000
    got = np.array([[o["probs"][c] for c in head["classes"]] for o in out])
    if name == "run_pass":
        expected = np.column_stack([1 - ref["p_pass"], ref["p_pass"]])
    else:
        expected = ref[[f"{name}_{c}" for c in head["classes"]]].to_numpy()
    diff = np.abs(got - expected).max()
    agree = (got.argmax(1) == expected.argmax(1)).mean()
    kb = os.path.getsize(paths[name]) / 1024
    print(f"{name:<10}{diff:>12.2e}{agree:>14.2%}{kb:>10.0f}{ms:>9.3f}")
    ok &= diff <= TOL

print(f"\n{'PASS' if ok else 'FAIL'}: every probability within {TOL:g} of Python")
raise SystemExit(0 if ok else 1)
