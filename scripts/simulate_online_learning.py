"""Does the app's online correction layer help? Offline simulation on unseen plays (2024-2025).

The frozen v3 run/pass model gives p_base. A tiny logistic regression on top (16 inputs, see FEATURES)
starts as the identity (p_adj = p_base) and takes one SGD step each time the user reveals a play's answer.
Scoring is prequential: each play is scored BEFORE the model learns from it, like in the app.

Run: uv run python -m scripts.simulate_online_learning
Writes app/Vision/Learning/learner_config.json and app/VisionTests/Fixtures/learner_sequence.json.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.prediction import model as m
from pipeline.prediction import predictor

FEATURES = ["bias", "base_logit", "down_1", "down_2", "down_3", "down_4", "ytg", "ytg_known", "field",
            "form_shotgun", "form_under_center", "form_pistol", "empty", "box", "score", "late"]
CLIP_LOGIT = 8.0
W0 = np.eye(len(FEATURES))[1]  # weight 1 on base_logit, 0 elsewhere: starts as "trust the base model"
LRS, L2S = [0.003, 0.01, 0.03, 0.1], [0, 0.001, 0.01, 0.1]
SEEDS = range(5)


def learner_x(df, p_base):
    """The 16-feature vector, exactly as the Swift learner builds it. Unknown inputs become 0."""
    p = np.clip(p_base, 1e-12, 1 - 1e-12)
    num = lambda c: df[c].astype(float).to_numpy()  # nullable ints/None -> NaN
    down, ytg, field, box, score = (num(c) for c in ["down", "ydstogo", "yardline_100", "ftn_box", "score_differential"])
    cols = [np.ones(len(df)), np.clip(np.log(p / (1 - p)), -CLIP_LOGIT, CLIP_LOGIT)]
    cols += [(down == d).astype(float) for d in (1, 2, 3, 4)]  # NaN == d is False -> all 0 when unknown
    cols += [np.nan_to_num(np.minimum(ytg, 20) / 10), (~np.isnan(ytg)).astype(float), np.nan_to_num(field / 100)]
    cols += [df["formation_family"].isin([f]).to_numpy(float) for f in ("shotgun", "under_center", "pistol")]
    cols += [(num("is_empty") == 1).astype(float), np.nan_to_num((box - 6.5) / 2),
             np.nan_to_num(np.clip(score, -21, 21) / 14), (num("qtr") >= 4).astype(float)]
    return np.column_stack(cols)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def logloss(p, y):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def run_one(x, y, lr, l2):
    """Plain one-play-at-a-time learner: the reference the Swift code must match."""
    w, p_before = W0.copy(), []
    for xi, yi in zip(x, y):
        p = sigmoid(w @ xi)
        p_before.append(p)
        w = w - lr * ((p - yi) * xi + l2 * (w - W0))
    return np.array(p_before), w


def run_grid(X, Y):
    """Same learner for every (seed-order, lr, l2) at once. X: (orders, n, 16), Y: (orders, n).
    Returns p_adj before each update, shape (orders, configs, n)."""
    lr = np.repeat(LRS, len(L2S))[None, :, None]
    l2 = np.tile(L2S, len(LRS))[None, :, None]
    W = np.broadcast_to(W0, (X.shape[0], lr.shape[1], len(W0))).copy()
    P = np.empty((X.shape[0], lr.shape[1], X.shape[1]))
    for t in range(X.shape[1]):
        x = X[:, t][:, None, :]
        p = sigmoid((W * x).sum(-1))
        P[:, :, t] = p
        W -= lr * ((p - Y[:, t][:, None])[..., None] * x + l2 * (W - W0))
    return P


df = pd.read_parquet("data/processed/plays_2016_2025.parquet")
df = df[df.season >= 2024].sort_values(["season", "week", "game_id", "play_id"]).reset_index(drop=True)
bundle = predictor.load()
p_base = predictor.predict_frame(bundle, df[m.V3_INPUTS])["p_pass"].to_numpy()
X_all, y_all = learner_x(df, p_base), df["is_pass"].to_numpy(float)
print(f"{len(df)} plays (2024-2025), base accuracy {((p_base >= 0.5) == y_all).mean():.4f}")

configs = [(lr, l2) for lr in LRS for l2 in L2S]
rows = []


def score(P, idx, case, n):
    """Mean over orders of prequential accuracy/log loss on the first n plays, base vs adjusted."""
    y, pb = y_all[idx[:, :n]], p_base[idx[:, :n]]
    base_ll, base_acc = logloss(pb, y).mean(), ((pb >= 0.5) == y).mean()
    for c, (lr, l2) in enumerate(configs):
        pa = P[:, c, :n]
        d_ll = logloss(pa, y).mean(1) - logloss(pb, y).mean(1)  # per order
        rows.append({"case": case, "n": n, "lr": lr, "l2": l2, "base_ll": base_ll, "adj_ll": logloss(pa, y).mean(),
                     "d_ll": d_ll.mean(), "d_ll_worst": d_ll.max(), "base_acc": base_acc,
                     "adj_acc": ((pa >= 0.5) == y).mean()})


# (a) random order over all plays, 5 seeds; (c) is the first 50 / 200 plays of those same orders.
rand = np.stack([np.random.default_rng(s).permutation(len(df)) for s in SEEDS])
P = run_grid(X_all[rand], y_all[rand])
score(P, rand, "a_random_all", len(df))
score(P, rand, "c_first", 200)
score(P, rand, "c_first", 50)
# (b) chronological order, one pass.
chrono = np.arange(len(df))[None]
score(run_grid(X_all[chrono], y_all[chrono]), chrono, "b_chronological", len(df))
# Extra noise check for the short cases: 100 random orders of the first 200 plays.
many = np.stack([np.random.default_rng(1000 + s).permutation(len(df))[:200] for s in range(100)])
P100 = run_grid(X_all[many], y_all[many])
score(P100, many, "c_first_100seeds", 200)
score(P100, many, "c_first_100seeds", 50)

res = pd.DataFrame(rows)
pd.set_option("display.width", 200)
for (case, n), g in res.groupby(["case", "n"], sort=False):
    print(f"\n{case} n={n}  base ll {g.base_ll.iloc[0]:.4f} acc {g.base_acc.iloc[0]:.4f}   (d_ll = adjusted - base; < 0 helps)")
    print(g.pivot(index="lr", columns="l2", values="d_ll").round(4).to_string())

# Pick: log loss may not get worse by more than 0.002 on the first 50 plays (mean over the 100 orders and
# over the 5 seeds), then the biggest gain at 200 plays and over all plays (random order).
key = ["lr", "l2"]
d = lambda case, n: res[(res.case == case) & (res.n == n)].set_index(key)["d_ll"]
table = pd.DataFrame({"d50_5": d("c_first", 50), "d50_100": d("c_first_100seeds", 50),
                      "d200_100": d("c_first_100seeds", 200), "d_all": d("a_random_all", len(df)),
                      "d_chrono": d("b_chronological", len(df))})
ok = table[(table.d50_5 <= 0.002) & (table.d50_100 <= 0.002)]
best_lr, best_l2 = (ok.d200_100 + ok.d_all).idxmin()
print(f"\nchosen lr={best_lr}, l2={best_l2}\n{table.round(4).to_string()}")

# --- Outputs for the app ---
config = {"features": FEATURES, "w0": W0.tolist(), "lr": best_lr, "l2": best_l2, "clip_logit": CLIP_LOGIT}
Path("app/Vision/Learning").mkdir(parents=True, exist_ok=True)
Path("app/Vision/Learning/learner_config.json").write_text(json.dumps(config, indent=2) + "\n")

# Fixture: 30 plays in random order, a few inputs blanked so the "unknown -> 0" rules get exercised.
fix = df.iloc[np.random.default_rng(7).permutation(len(df))[:30]].reset_index(drop=True)
inputs = fix[m.V3_INPUTS].astype(object).where(fix[m.V3_INPUTS].notna(), None)
blanks = {2: ["down"], 5: ["ydstogo"], 8: ["formation_family", "is_empty"], 11: ["ftn_box"],
          14: ["score_differential", "qtr"], 17: ["yardline_100"], 20: m.V3_INPUTS}
for i, cols in blanks.items():
    inputs.loc[i, cols] = None
pb_fix = predictor.predict_frame(bundle, inputs.astype({c: float for c in m.V3_INPUTS
                                                        if c not in m.CATEGORIES}))["p_pass"].to_numpy()
x_fix = learner_x(inputs.astype({c: float for c in m.V3_INPUTS if c not in m.CATEGORIES}), pb_fix)
y_fix = fix["is_pass"].to_numpy(float)
p_fix, w_final = run_one(x_fix, y_fix, best_lr, best_l2)
# Cross-check: the vectorised grid simulator gives the same numbers as the plain loop.
c = configs.index((best_lr, best_l2))
assert np.allclose(run_grid(x_fix[None], y_fix[None])[0, c], p_fix, atol=1e-12)
keep = ["down", "ydstogo", "yardline_100", "formation_family", "is_empty", "ftn_box", "score_differential", "qtr"]
plays = [{"inputs": {k: (None if inputs.at[i, k] is None else (inputs.at[i, k] if k == "formation_family"
                                                                else float(inputs.at[i, k]))) for k in keep},
          "p_base": float(pb_fix[i]), "x": x_fix[i].tolist(), "y": int(y_fix[i]), "p_adj_before": float(p_fix[i])}
         for i in range(30)]
fixture = {"features": FEATURES, "w0": W0.tolist(), "lr": best_lr, "l2": best_l2, "clip_logit": CLIP_LOGIT,
           "plays": plays, "w_final": w_final.tolist()}
Path("app/VisionTests/Fixtures/learner_sequence.json").write_text(json.dumps(fixture, indent=2) + "\n")
Path("models/metrics_online_learning.json").write_text(json.dumps(
    {"chosen": {"lr": best_lr, "l2": best_l2}, "rows": res.to_dict("records")}, indent=2, default=float))
print("\nwrote app/Vision/Learning/learner_config.json, app/VisionTests/Fixtures/learner_sequence.json, "
      "models/metrics_online_learning.json")
