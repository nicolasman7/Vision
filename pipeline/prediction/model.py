"""Pre-snap play predictor: run/pass and direction, with honest (time-split) evaluation."""

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

# Game situation: what's on the scoreboard and the down marker.
SITUATION = [
    "down", "ydstogo", "yardline_100", "qtr",
    "quarter_seconds_remaining", "half_seconds_remaining", "game_seconds_remaining",
    "score_differential", "posteam_timeouts_remaining", "defteam_timeouts_remaining", "goal_to_go",
]
# What the camera can see before the snap: formation, who's on the field, how many in the box.
# The raw personnel strings are left out; off_*/def_* are the same info as numbers.
ALIGNMENT = [
    "shotgun", "no_huddle", "offense_formation", "defenders_in_box", "n_offense", "n_defense",
    "off_ol", "off_rb", "off_te", "off_wr", "def_dl", "def_lb", "def_db",
]
ALL_FEATURES = SITUATION + ALIGNMENT
# Direction model also gets is_pass: in the app the run/pass call is made first, then direction.
DIRECTION_FEATURES = ALL_FEATURES + ["is_pass"]
LABELS = ["is_pass", "direction", "air_yards"]

# Fixed category list, so the integer codes the model learned mean the same thing at prediction time.
FORMATIONS = ["SHOTGUN", "SINGLEBACK", "EMPTY", "I_FORM", "PISTOL", "JUMBO", "WILDCAT"]


def time_split(df, last_train_week=14):
    """Train on early weeks, test on later ones: mimics predicting games that haven't happened yet.

    A random split would let the model see Week 16 Chiefs plays while testing on other Week 16 Chiefs
    plays, which flatters it. Splitting by week also keeps every game entirely on one side.
    """
    return df[df["week"] <= last_train_week], df[df["week"] > last_train_week]


def features(df, cols):
    """Select columns in a form the model accepts: floats (NaN = missing) and one categorical."""
    X = df[cols].copy()
    for c in cols:
        if c == "offense_formation":
            X[c] = pd.Categorical(X[c], categories=FORMATIONS)
        else:
            X[c] = X[c].astype("float64")  # nullable Int64 -> float with NaN, which HGB handles natively
    return X


def _hgb():
    # Library defaults are sensible for ~25k rows; early stopping holds out 10% of the training
    # rows and stops adding trees once that score stops improving, which guards against overfitting.
    # categorical_features="from_dtype" treats the pandas Categorical column as categories, not numbers.
    return HistGradientBoostingClassifier(early_stopping=True, categorical_features="from_dtype", random_state=0)


def train_run_pass(train, cols):
    return _hgb().fit(features(train, cols), train["is_pass"])


def train_direction(train):
    known = train[train["direction"].notna()]
    return _hgb().fit(features(known, DIRECTION_FEATURES), known["direction"].astype(str))


def calibrate(model, calib, cols):
    """Isotonic recalibration on held-out weeks: learns a monotone map from 'model says p' to 'actually p'.

    FrozenEstimator keeps the already-trained model fixed; only the probability mapping is fit.
    """
    return CalibratedClassifierCV(FrozenEstimator(model), method="isotonic").fit(
        features(calib, cols), calib["is_pass"])


def evaluate(y, p_pass):
    """Binary metrics. Log loss and Brier score judge the probabilities, not just the hard call."""
    return {
        "accuracy": accuracy_score(y, p_pass >= 0.5),
        # Log loss punishes confident mistakes hard (being 99% sure and wrong costs a lot).
        "log_loss": log_loss(y, p_pass, labels=[0, 1]),
        # Brier = mean squared error between probability and the 0/1 outcome. Lower is better.
        "brier": brier_score_loss(y, p_pass),
    }


def calibration_table(y, p_pass, edges=(0.5, 0.6, 0.7, 0.8, 0.9, 1.0)):
    """Group predictions by confidence (prob of the predicted class) and check how often each group is right.

    Well calibrated = in the 80-90% bin the model is right about 85% of the time.
    """
    y, p_pass = np.asarray(y), np.asarray(p_pass)
    conf = np.maximum(p_pass, 1 - p_pass)
    correct = (p_pass >= 0.5) == y
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & ((conf < hi) if hi < 1 else (conf <= hi))
        rows.append({"bin": f"{lo:.0%}-{hi:.0%}", "count": int(m.sum()),
                     "mean_confidence": float(conf[m].mean()) if m.any() else None,
                     "accuracy": float(correct[m].mean()) if m.any() else None})
    return rows
