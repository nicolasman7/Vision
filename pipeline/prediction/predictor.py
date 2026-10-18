"""Use the v3 predictor bundle: any subset of inputs in, calibrated probabilities out.

    bundle = load("models/predictor_v3.joblib")
    predict(bundle, down=3, ydstogo=8, formation_family="shotgun")
    -> {"p_pass": 0.9, "direction": {"left": ..., "middle": ..., "right": ...}, "depth": {"<=0": ..., ...}}

Anything not passed is unknown (NaN); the model was trained with blanked inputs, so it still answers.
"""

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, softmax

from pipeline.prediction.model import onehot


def load(path="models/predictor_v3.joblib"):
    return joblib.load(path)


def derive(df):
    """Fill the "derived" inputs from the ones the user gives, when they weren't passed directly.

    Checked against the data: these formulas match nflverse on every play (overtime counts as its own half).
    """
    df = df.copy()
    gtg = (df["ydstogo"] == df["yardline_100"]).astype("float64")  # goal to go: the first-down line is the goal line
    df["goal_to_go"] = df["goal_to_go"].fillna(gtg.where(df["ydstogo"].notna() & df["yardline_100"].notna()))
    q, qs = df["qtr"], df["quarter_seconds_remaining"]
    half = (qs + np.where(q.isin([1, 3]), 900, 0)).where(q.notna())  # Q1/Q3 still have a full quarter to go
    df["half_seconds_remaining"] = df["half_seconds_remaining"].fillna(half)
    df["game_seconds_remaining"] = df["game_seconds_remaining"].fillna(qs + 900 * (4 - q).clip(lower=0))
    return df


def matrix(bundle, df):
    """Raw inputs -> the exact numeric feature vector the trees (and the Core ML model) expect."""
    inputs = list(bundle["sources"])
    unknown = set(df.columns) - set(inputs)
    if unknown:
        raise ValueError(f"unknown inputs {sorted(unknown)}; expected some of {inputs}")
    df = df.reindex(columns=inputs).astype({c: "object" for c in bundle["onehot"]})
    num = [c for c in inputs if c not in bundle["onehot"]]
    df[num] = df[num].astype("float64")  # None -> NaN
    return onehot(derive(df), inputs)[bundle["features"]]


def predict_frame(bundle, df):
    """Many plays at once. One row per play: p_pass, direction_<class> and depth_<class> probabilities."""
    X = matrix(bundle, df)
    out = {}
    for name, head in bundle["heads"].items():
        raw = head["model"].decision_function(X)
        if name == "run_pass":
            out["p_pass"] = expit(head["calib"]["a"] * raw + head["calib"]["b"])
        else:
            p = softmax(raw / head["calib"]["T"], axis=1)
            for i, c in enumerate(head["classes"]):
                out[f"{name}_{c}"] = p[:, i]
    return pd.DataFrame(out, index=df.index)


def predict(bundle, **inputs):
    """One play. Direction assumes the play is a run, depth assumes the ball is thrown; both always come back."""
    row = predict_frame(bundle, pd.DataFrame([inputs]) if inputs else pd.DataFrame(index=[0])).iloc[0]
    return {"p_pass": float(row["p_pass"]),
            **{h: {c: float(row[f"{h}_{c}"]) for c in bundle["heads"][h]["classes"]} for h in ("direction", "depth")}}
