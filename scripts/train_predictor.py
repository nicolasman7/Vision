"""Train the app-ready v3 predictor bundle and report on 2025. Run: uv run python -m scripts.train_predictor

Train 2016-2023, fit calibration on 2024, test on 2025 (calibration never sees 2025).
Writes models/predictor_v3.joblib and models/metrics_v3.json.
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from pipeline.prediction import model as m
from pipeline.prediction import predictor

df = pd.read_parquet("data/processed/plays_2016_2025.parquet")
train, calib, test = df[df.season <= 2023], df[df.season == 2024], df[df.season == 2025]
print(f"train 2016-2023: {len(train)}  calibration 2024: {len(calib)}  test 2025: {len(test)}")
results = {"split": {"train": len(train), "calib": len(calib), "test": len(test)},
           "dropout": m.DROPOUT, "group_dropout": m.GROUP_DROPOUT}

bundle = m.train_v3(train, calib)  # with input dropout: the one the app gets
plain = m.train_v3(train, calib, dropout=False)  # same thing without dropout, for comparison
# Run/pass ships uncalibrated (identity): the Platt fit on 2024 made 2025 worse, and the one-season-earlier
# backtest (section e) showed it was a no-op. The fit is kept for the record. Both bundles, so (c)/(d) compare alike.
for b in (bundle, plain):
    rp = b["heads"]["run_pass"]
    rp["calib_fitted_2024"], rp["calib"] = rp["calib"], {"a": 1.0, "b": 0.0}
joblib.dump(bundle, "models/predictor_v3.joblib")


def scores(y, P):
    P = np.asarray(P)
    return {"accuracy": float((P.argmax(1) == y).mean()), "log_loss": float(log_loss(y, P, labels=range(P.shape[1])))}


def two_col(p):
    return np.column_stack([1 - p, p])


def blank(df, cols, frac=1.0, seed=0):
    """Set cols to NaN on a random `frac` of rows (all rows by default)."""
    df = df.copy()
    df.loc[np.random.default_rng(seed).random(len(df)) < frac, cols] = np.nan
    return df


def p_pass(b, inputs, calibrated=True):
    if calibrated:
        return predictor.predict_frame(b, inputs)["p_pass"].to_numpy()
    return b["heads"]["run_pass"]["model"].predict_proba(predictor.matrix(b, inputs))[:, 1]


def show(title, table):
    print(f"\n{title}\n{pd.DataFrame(table).T.round(4).to_string()}")


# --- (a) Each head vs its base-rate baseline on 2025 with full inputs, calibration before/after ---
results["a_heads"] = {}
for head, classes in m.HEADS.items():
    rows, y = m.head_rows(test, head)
    _, y_train = m.head_rows(train, head)
    X = predictor.matrix(bundle, rows[m.V3_INPUTS])
    h = bundle["heads"][head]
    raw_p = h["model"].predict_proba(X)  # before calibration
    out = predictor.predict_frame(bundle, rows[m.V3_INPUTS])
    cal_p = two_col(out["p_pass"].to_numpy()) if head == "run_pass" else out[[f"{head}_{c}" for c in classes]].to_numpy()
    base = np.tile(np.bincount(y_train, minlength=len(classes)) / len(y_train), (len(y), 1))
    table = {"base_rate": scores(y, base), "model_raw": scores(y, raw_p), "model_calibrated": scores(y, cal_p)}
    if head == "run_pass":
        before, after = m.calibration_table(y, raw_p[:, 1]), m.calibration_table(y, cal_p[:, 1])
    else:
        before, after = m.calibration_table_multiclass(y, raw_p), m.calibration_table_multiclass(y, cal_p)
    results["a_heads"][head] = {**table, "test_rows": len(y), "calib": h["calib"],
                                "calibration_before": before, "calibration_after": after}
    show(f"(a) {head}  ({len(y)} test plays, calib {h['calib']})", table)
    print(pd.DataFrame({"bin": [r["bin"] for r in before], "count": [r["count"] for r in before],
                        "conf_before": [r["mean_confidence"] for r in before],
                        "acc_before": [r["accuracy"] for r in before],
                        "conf_after": [r["mean_confidence"] for r in after],
                        "acc_after": [r["accuracy"] for r in after]}).round(3).to_string(index=False))

# --- (b) Availability patterns: one dropout model vs a dedicated no-dropout model per pattern (run/pass) ---
y = test["is_pass"].to_numpy()
test_in = test[m.V3_INPUTS]
patterns = {
    "full": [],
    "no_situation (alignment only)": m.SITUATION,
    "user_input (down & distance + alignment)": [c for c in m.SITUATION if c not in ("down", "ydstogo")],
    "situation_only (vision failed)": m.V3_VISION,
    "nothing": m.V3_INPUTS,
}
results["b_patterns"] = {}
for name, cols in patterns.items():
    if name == "full":
        p_ded = p_pass(plain, test_in, calibrated=False)
    elif name == "nothing":
        p_ded = np.full(len(y), train["is_pass"].mean())
    else:  # trained on exactly the inputs this pattern has (HGB can't take all-NaN columns, so drop them)
        keep = [c for c in m.V3_INPUTS if c not in cols]
        ded = m._hgb_plain().fit(m.onehot(train, keep), train["is_pass"])
        p_ded = ded.predict_proba(m.onehot(test_in, keep))[:, 1]
    t = blank(test_in, cols)
    results["b_patterns"][name] = {
        "dedicated": scores(y, two_col(p_ded)),
        "dropout_raw": scores(y, two_col(p_pass(bundle, t, calibrated=False))),
        "dropout_calibrated": scores(y, two_col(p_pass(bundle, t))),
    }
print("\n(b) run/pass by input availability, 2025")
print(pd.DataFrame({k: {f"{mod} {s}": v[mod][s] for mod in v for s in v[mod]} for k, v in results["b_patterns"].items()})
      .T.round(4).to_string())

# --- (c) Wrong vs blank formation_family, for the no-dropout and dropout models ---
train_ff = train["formation_family"]
ff = test_in.assign(formation_family=pd.Categorical(test_in["formation_family"], categories=m.CATEGORIES["formation_family"]))
results["c_wrong_vs_blank"] = {}
for frac in [0.0, 0.1, 0.2, 0.3]:
    wrong = m.corrupt_category(ff, "formation_family", frac, train_ff)
    empty = blank(test_in, ["formation_family"], frac)
    row = {}
    for label, b in [("no_dropout", plain), ("dropout", bundle)]:
        for kind, inputs in [("wrong", wrong), ("blank", empty)]:
            s = scores(y, two_col(p_pass(b, inputs)))
            row[f"{label} {kind} acc"], row[f"{label} {kind} ll"] = s["accuracy"], s["log_loss"]
    results["c_wrong_vs_blank"][f"{frac:.0%}"] = row
show("(c) formation_family misread vs left blank (calibrated, 2025)", results["c_wrong_vs_blank"])

# --- (d) Clean-data cost of dropout: full inputs, calibrated ---
results["d_clean_cost"] = {"no_dropout": scores(y, two_col(p_pass(plain, test_in))),
                           "dropout": scores(y, two_col(p_pass(bundle, test_in)))}
show("(d) full inputs, calibrated, 2025", results["d_clean_cost"])

# --- (e) Does Platt scaling fit on the previous season help? Backtest one year earlier, so the answer
# doesn't come from peeking at 2025: train 2016-2022, calibrate on 2023, test on 2024. ---
bt = m.train_v3(df[df.season <= 2022], df[df.season == 2023])
d24 = df[df.season == 2024]
y24, in24 = d24["is_pass"].to_numpy(), d24[m.V3_INPUTS]
p_raw24, p_cal24 = p_pass(bt, in24, calibrated=False), p_pass(bt, in24)
results["e_platt_backtest_2024"] = {
    "calib": bt["heads"]["run_pass"]["calib"],
    "raw": {**scores(y24, two_col(p_raw24)), "table": m.calibration_table(y24, p_raw24)},
    "platt": {**scores(y24, two_col(p_cal24)), "table": m.calibration_table(y24, p_cal24)},
}
show(f"(e) Platt backtest: train 2016-2022, calib 2023 {bt['heads']['run_pass']['calib']}, test 2024",
     {k: {c: v for c, v in r.items() if c != "table"} for k, r in results["e_platt_backtest_2024"].items() if k != "calib"})
print(pd.DataFrame({k: [f"{r['mean_confidence']:.3f}/{r['accuracy']:.3f}" for r in v["table"]]
                    for k, v in results["e_platt_backtest_2024"].items() if k != "calib"},
                   index=[r["bin"] for r in results["e_platt_backtest_2024"]["raw"]["table"]]).to_string())

results["bundle"] = {"path": "models/predictor_v3.joblib", "features": bundle["features"],
                     "onehot": bundle["onehot"], "sources": bundle["sources"],
                     "heads": {h: {k: v[k] for k in v if k != "model"} for h, v in bundle["heads"].items()},
                     "note": "run_pass calib is identity (a=1, b=0) by decision; calib_fitted_2024 kept for the record"}
with open("models/metrics_v3.json", "w") as f:
    json.dump(results, f, indent=2, default=float)
print("\nsaved models/predictor_v3.joblib, models/metrics_v3.json")
