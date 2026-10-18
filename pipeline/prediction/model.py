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
LABELS = ["is_pass", "direction", "air_yards", "run_gap", "pass_length"]

# Fixed category list, so the integer codes the model learned mean the same thing at prediction time.
FORMATIONS = ["SHOTGUN", "SINGLEBACK", "EMPTY", "I_FORM", "PISTOL", "JUMBO", "WILDCAT"]
CATEGORIES = {
    "offense_formation": FORMATIONS,
    "formation_family": ["shotgun", "under_center", "pistol"],
    "ftn_starting_hash": ["L", "M", "R"],
    "ftn_qb_location": ["U", "S", "P"],  # under center / shotgun / pistol
}

# Multi-season features. offense_formation (the 7 NGS labels) stops in 2022, so models meant to work on
# 2023+ use formation_family + is_empty instead. Each box count stays in its own era column (NaN outside
# its era): the sources count "the box" differently, and merging them would blur the meaning.
PERSONNEL = ["off_ol", "off_rb", "off_te", "off_wr", "def_dl", "def_lb", "def_db"]
ERA_ALIGNMENT = ["shotgun", "no_huddle", "formation_family", "is_empty", "n_offense", "n_defense"] + PERSONNEL \
    + ["defenders_in_box", "box_part23"]
# FTN hand-charted pre-snap details, 2022+ only (NaN before).
FTN = ["ftn_box", "ftn_starting_hash", "ftn_qb_location", "ftn_n_offense_backfield"]
# Borderline: some motion happens at the snap, which a pre-snap read might not see.
FTN_MOTION = ["ftn_is_motion"]
ERA_FEATURES = SITUATION + ERA_ALIGNMENT + FTN


def time_split(df, last_train_week=14):
    """Train on early weeks, test on later ones: mimics predicting games that haven't happened yet.

    A random split would let the model see Week 16 Chiefs plays while testing on other Week 16 Chiefs
    plays, which flatters it. Splitting by week also keeps every game entirely on one side.
    """
    return df[df["week"] <= last_train_week], df[df["week"] > last_train_week]


def season_split(df, test_season, last_train_week=0, train_seasons=None):
    """Test = one season after week last_train_week; train = everything strictly before that point.

    train_seasons optionally restricts which earlier seasons are used (e.g. only recent ones, in case
    old seasons hurt because the game has drifted). Training never sees a later (season, week).
    """
    test = (df["season"] == test_season) & (df["week"] > last_train_week)
    before = (df["season"] < test_season) | ((df["season"] == test_season) & (df["week"] <= last_train_week))
    if train_seasons is not None:
        before &= df["season"].isin(train_seasons)
    return df[before], df[test]


def features(df, cols):
    """Select columns in a form the model accepts: floats (NaN = missing) and fixed-list categoricals."""
    X = df[cols].copy()
    for c in cols:
        if c in CATEGORIES:
            X[c] = pd.Categorical(X[c], categories=CATEGORIES[c])  # values outside the list -> NaN
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


# --- Simulated vision errors: the app reads formation and box count from video, and it will be wrong sometimes.

def corrupt_category(X, col, frac, train_values, seed=0):
    """Misread `frac` of rows: each picked row gets a different value, drawn from the training distribution.

    Drawing from training frequencies mimics a detector that mostly confuses things with common formations.
    """
    rng = np.random.default_rng(seed)
    X = X.copy()
    freq = pd.Series(train_values).dropna().astype(str).value_counts(normalize=True)
    choices, probs = freq.index.to_numpy(), freq.to_numpy()
    col_values = np.array(X[col].astype(object))  # writable copy (pandas may hand back a read-only view)
    rows = np.flatnonzero(rng.random(len(X)) < frac)
    new = rng.choice(choices, size=len(rows), p=probs)
    # Redraw until every picked row actually changed (a "misread" that equals the truth isn't an error).
    same = new == col_values[rows]
    while same.any():
        new[same] = rng.choice(choices, size=same.sum(), p=probs)
        same = new == col_values[rows]
    col_values[rows] = new
    X[col] = pd.Categorical(col_values, categories=X[col].cat.categories)  # X comes from features(), so col is categorical
    return X


def shift_count(X, col, frac, seed=0):
    """Miscount `frac` of rows by one player (+1 or -1 at random), never below 0. NaN stays NaN."""
    rng = np.random.default_rng(seed)
    X = X.copy()
    pick = rng.random(len(X)) < frac
    X.loc[pick, col] = (X.loc[pick, col] + rng.choice([-1, 1], size=pick.sum())).clip(lower=0)
    return X


# --- v3: one app-ready predictor (run/pass, run direction, throw depth) that copes with missing inputs.

# The app's inputs. Left out on purpose: shotgun and ftn_qb_location (same information as formation_family,
# so the app would have to fill the same thing twice), n_offense/n_defense (always 11), and the other box columns.
# Box: ftn_box is the one box count, because it is a single definition (FTN charting) from 2022 to today.
# defenders_in_box (NGS, <=2022) agrees with FTN on only 59% of 2022 plays and box_part23 starts in 2023, so
# mixing them would make "box" mean different things in different rows. Before 2022 ftn_box is simply NaN.
V3_VISION = ["no_huddle", "formation_family", "is_empty", "ftn_n_offense_backfield"] + PERSONNEL \
    + ["ftn_box", "ftn_starting_hash"]
V3_INPUTS = SITUATION + V3_VISION

# Where the app gets each input. "user" = typed in the situation panel (or a scorebug reader later),
# "vision" = read from the video, "derived" = computed from other inputs (see predictor.derive).
FEATURE_SOURCES = {c: "user" for c in SITUATION} | {c: "vision" for c in V3_VISION} | {
    "goal_to_go": "derived", "half_seconds_remaining": "derived", "game_seconds_remaining": "derived"}
FEATURE_SOURCES = {c: FEATURE_SOURCES[c] for c in V3_INPUTS}  # keep the input order

# Input blocks that tend to be missing together. goal_to_go needs both ydstogo and yardline_100, so it
# sits in both blocks and is blanked when either one is.
BLOCKS = {
    "down_distance": ["down", "ydstogo", "goal_to_go"],
    "field_position": ["yardline_100", "goal_to_go"],
    "scoreboard": ["score_differential", "qtr", "quarter_seconds_remaining", "half_seconds_remaining",
                   "game_seconds_remaining", "posteam_timeouts_remaining", "defteam_timeouts_remaining"],
    # no_huddle rides with formation: both come from the same pre-snap look at the offense.
    "formation": ["formation_family", "is_empty", "ftn_n_offense_backfield", "no_huddle"],
    "personnel": PERSONNEL,
    "box": ["ftn_box"],
    "hash": ["ftn_starting_hash"],
}
# Chance that each block is blanked on a dropout row, picked to mirror how the app will be used:
# down & distance is the first thing the user enters (rarely missing); the scoreboard needs typing or a
# scorebug reader (often missing); personnel and box need a clear wide view (missing a fair amount);
# formation is the vision stage's main output (usually readable).
DROPOUT = {"down_distance": 0.15, "field_position": 0.3, "scoreboard": 0.4,
           "formation": 0.15, "personnel": 0.3, "box": 0.3, "hash": 0.2}
# On top of that, whole halves go missing together: the user hasn't entered anything yet (no situation),
# or the camera view was useless (no vision). Independent blocks alone would almost never produce these.
GROUP_DROPOUT = {"situation": 0.25, "vision": 0.1}
GROUPS = {"situation": ["down_distance", "field_position", "scoreboard"],
          "vision": ["formation", "personnel", "box", "hash"]}

HEADS = {"run_pass": [0, 1], "direction": ["left", "middle", "right"], "depth": ["<=0", "1-9", "10+"]}


def onehot(df, cols):
    """Like features(), but categories become one float column per value (`col=value`) instead of a
    pandas categorical: Core ML can't export HGB's categorical splits. A missing (or unknown) value
    is NaN in every one of its columns, so the trees learn a "missing" branch rather than reading 0 as "no"."""
    out = {}
    for c in cols:
        if c in CATEGORIES:
            known = df[c].isin(CATEGORIES[c])
            for v in CATEGORIES[c]:
                out[f"{c}={v}"] = df[c].isin([v]).astype("float64").where(known)
        else:
            out[c] = df[c].astype("float64")
    return pd.DataFrame(out, index=df.index)


def apply_dropout(df, seed=0):
    """Copy of df with whole input blocks blanked per row (rates in DROPOUT and GROUP_DROPOUT)."""
    rng = np.random.default_rng(seed)
    df = df.copy()
    group_off = {g: rng.random(len(df)) < p for g, p in GROUP_DROPOUT.items()}
    for block, cols in BLOCKS.items():
        off = rng.random(len(df)) < DROPOUT[block]
        for g, blocks in GROUPS.items():
            if block in blocks:
                off |= group_off[g]
        df.loc[off, cols] = np.nan
    return df


def head_rows(df, head):
    """Rows a head learns from, and their labels as class indices into HEADS[head]."""
    if head == "run_pass":
        return df, df["is_pass"].to_numpy()
    if head == "direction":  # runs with a known direction
        runs = df[(df["is_pass"] == 0) & df["direction"].notna()]
        return runs, runs["direction"].map({d: i for i, d in enumerate(HEADS["direction"])}).to_numpy()
    # depth: throws only. Sacks and scrambles have no air_yards, so they drop out here.
    throws = df[(df["is_pass"] == 1) & df["air_yards"].notna()]
    ay = throws["air_yards"].to_numpy()
    return throws, np.select([ay <= 0, ay < 10], [0, 1], 2)


def fit_platt(raw, y):
    """Fit p = sigmoid(a*raw + b) by minimising log loss. Two numbers, so the exporter can fold them into the trees."""
    from scipy.optimize import minimize

    def nll(ab):
        z = ab[0] * raw + ab[1]
        return np.mean(np.logaddexp(0, z) - y * z)
    a, b = minimize(nll, [1.0, 0.0]).x
    return {"a": float(a), "b": float(b)}


def fit_temperature(raw, y):
    """Fit p = softmax(raw / T). T > 1 softens overconfident probabilities, T < 1 sharpens them."""
    from scipy.optimize import minimize_scalar
    from scipy.special import logsumexp

    def nll(T):
        z = raw / T
        return np.mean(logsumexp(z, axis=1) - z[np.arange(len(y)), y])
    return {"T": float(minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x)}


def calibration_table_multiclass(y, proba, edges=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0)):
    """calibration_table for 3 classes: bin by the top probability, check how often the top class is right."""
    proba, y = np.asarray(proba), np.asarray(y)
    conf, correct = proba.max(axis=1), proba.argmax(axis=1) == y
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & ((conf < hi) if hi < 1 else (conf <= hi))
        rows.append({"bin": f"{lo:.0%}-{hi:.0%}", "count": int(m.sum()),
                     "mean_confidence": float(conf[m].mean()) if m.any() else None,
                     "accuracy": float(correct[m].mean()) if m.any() else None})
    return rows


def _hgb_plain():
    # No categorical splits (Core ML can't export them); categories arrive one-hot from onehot().
    return HistGradientBoostingClassifier(early_stopping=True, categorical_features=None, random_state=0)


def train_v3(train, calib, dropout=True):
    """Train the three heads and their calibration; return the app bundle (see scripts/train_predictor.py).

    With dropout, each set is the clean rows plus a copy with blocks blanked, so the model learns both
    "everything visible" and every kind of partial view. Calibration is fit on held-out seasons only.
    """
    if dropout:
        train = pd.concat([train, apply_dropout(train, seed=0)], ignore_index=True)
        calib = pd.concat([calib, apply_dropout(calib, seed=1)], ignore_index=True)
    heads = {}
    for head, classes in HEADS.items():
        rows, y = head_rows(train, head)
        model = _hgb_plain().fit(onehot(rows, V3_INPUTS), y)
        crows, cy = head_rows(calib, head)
        raw = model.decision_function(onehot(crows, V3_INPUTS))
        cal = fit_platt(raw, cy) if head == "run_pass" else fit_temperature(raw, cy)
        heads[head] = {"model": model, "classes": classes, "calib": cal}
    return {
        "version": 3,
        "features": list(onehot(train.head(1), V3_INPUTS).columns),
        "onehot": {c: CATEGORIES[c] for c in V3_INPUTS if c in CATEGORIES},
        "sources": FEATURE_SOURCES,
        "heads": heads,
    }
