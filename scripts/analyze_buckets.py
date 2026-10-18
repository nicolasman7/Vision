"""Compare candidate label schemes ("buckets") for play direction and pass depth.

Evidence only: the user picks the targets. Every scheme gets the same honest setup: time split,
training-frequency baseline, one HistGradientBoostingClassifier, accuracy + log loss + skill.

Run from repo root: uv run python -m scripts.analyze_buckets   ->  models/metrics_buckets.json
"""

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss

from pipeline.prediction import model as m

DATA = "data/processed/plays_2016_2025.parquet"
OUT = "models/metrics_buckets.json"
TEST_SEASONS = [2022, 2025]

# Pre-snap columns beyond model.ALL_FEATURES that cover every season (formation_family, is_empty)
# or 2022+ (FTN). HGB treats NaN as "missing", so older seasons simply lack the FTN values.
EXTRA = ["formation_family", "is_empty", "box_part23", "ftn_starting_hash", "ftn_qb_location",
         "ftn_n_offense_backfield", "ftn_box", "ftn_is_motion"]
CATEGORICAL = ["formation_family", "ftn_starting_hash", "ftn_qb_location"]
CATEGORIES = {}  # filled from the full table in main(), so train and test share category codes
TENDENCY = ["team_run_left", "team_run_middle", "team_pass_left", "team_pass_middle"]


def split(df, season, last_train_week=14):
    """Test on `season` weeks 15+; train on every earlier season plus weeks 1-14 of that season."""
    early = (df["season"] < season) | ((df["season"] == season) & (df["week"] <= last_train_week))
    return df[early], df[(df["season"] == season) & (df["week"] > last_train_week)]


def team_tendency(df):
    """Per play: the offense's left/middle share on runs and on passes, from its EARLIER weeks this season.

    Counting only weeks < w is what keeps this honest: a team's week-16 plays must not inform its
    week-16 feature. Teams play at most once a week, so "earlier weeks" = "earlier games".
    Week 1 has no history yet, so it gets NaN (HGB treats that as missing).
    """
    keys = ["season", "posteam", "week"]
    counts = df[keys].copy()
    for name, flag in [("run", 0), ("pass", 1)]:
        for side in ["left", "middle", "right"]:
            counts[f"{name}_{side}"] = ((df["is_pass"] == flag) & df["direction"].eq(side).fillna(False)).astype(int)
    week = counts.groupby(keys).sum()
    # Running total through week w, minus week w itself = total over weeks strictly before w.
    before = week.groupby(level=["season", "posteam"]).cumsum() - week
    out = pd.DataFrame(index=before.index)
    for name in ["run", "pass"]:
        n = before[[f"{name}_left", f"{name}_middle", f"{name}_right"]].sum(axis=1).replace(0, np.nan)
        out[f"team_{name}_left"] = before[f"{name}_left"] / n
        out[f"team_{name}_middle"] = before[f"{name}_middle"] / n
    return df[keys].join(out, on=keys)[TENDENCY]


def field_relative(direction, starting_hash):
    """left/middle/right -> toward the ball's hash side / middle / toward the wide side.

    Assumes FTN's L/R hash and nflverse's left/right share a frame (reading A). FTN doesn't say whose
    view its hash uses; if it's the other frame (reading B), hash_side and wide_side just swap names.
    Middle-hash snaps have no short side, so they get NaN.
    """
    hash_dir = starting_hash.map({"L": "left", "R": "right"})
    out = pd.Series(np.where(direction == hash_dir, "hash_side", "wide_side"), index=direction.index, dtype=object)
    out[direction == "middle"] = "middle"
    return out.where(direction.notna() & hash_dir.notna())


def labels(df):
    # object dtype: comparisons with a missing direction give False instead of <NA>.
    run, d = df["is_pass"] == 0, df["direction"].astype(object)
    passes = df["is_pass"] == 1
    gap = (d + "_" + df["run_gap"].astype("string")).where(run & df["run_gap"].notna())
    a = df["air_yards"]
    p2 = pd.cut(a, [-np.inf, 0, 9, 19, np.inf], labels=["behind", "short", "intermediate", "deep"]).astype(object)
    return {
        "D1": d, "D6": d,
        "D1p": d.where(~run),  # diagnostic: the pass half of D1, to compare with D2 (the run half)
        "D2": d.where(run),
        "D3": d.where(run & d.isin(["left", "right"])),
        "D4": gap.where(~(run & (d == "middle")), "middle"),
        "D5": field_relative(d, df["ftn_starting_hash"]),
        "P1": df["pass_length"].where(passes),
        "P2": p2.where(passes),
        # P3: behind / short / downfield (10+). Merges P2's two thinnest classes (13% and 12%).
        "P3": pd.cut(a, [-np.inf, 0, 9, np.inf], labels=["behind", "short", "downfield"]).astype(object).where(passes),
        # Sack/scramble options: give "no throw" its own class, or predict it on its own (yes/no).
        "P2_nothrow": p2.where(a.notna(), "no_throw").where(passes),
        "nothrow": a.isna().map({True: "no_throw", False: "throw"}).where(passes),
    }


# name -> (label key, extra feature columns, baseline grouping, first season with data, description)
DIRECTION_FEATS = m.ALL_FEATURES + EXTRA + ["is_pass"]
DEPTH_FEATS = m.ALL_FEATURES + EXTRA
SCHEMES = {
    "D1": (DIRECTION_FEATS, "is_pass", 2016, "L/M/R, all plays (today)"),
    "D1p": (DIRECTION_FEATS, None, 2016, "passes only, L/M/R (diagnostic)"),
    "D2": (DIRECTION_FEATS, None, 2016, "runs only, L/M/R"),
    "D3": (DIRECTION_FEATS, None, 2016, "runs only, L/R (middle dropped)"),
    "D4": (DIRECTION_FEATS, None, 2016, "runs by gap: end/tackle/guard x side + middle"),
    "D5": (DIRECTION_FEATS, "is_pass", 2022, "field-relative: hash side / middle / wide side (L/R hash only)"),
    "D6": (DIRECTION_FEATS + TENDENCY, "is_pass", 2016, "D1 + team tendency from earlier weeks"),
    "P1": (DEPTH_FEATS, None, 2016, "nflverse pass_length short/deep"),
    "P2": (DEPTH_FEATS, None, 2016, "air yards <=0 / 1-9 / 10-19 / 20+"),
    "P3": (DEPTH_FEATS, None, 2016, "air yards <=0 / 1-9 / 10+"),
    "P2_nothrow": (DEPTH_FEATS, None, 2016, "P2 + 'no_throw' class for sacks/scrambles"),
    "nothrow": (DEPTH_FEATS, None, 2016, "dropbacks: will the ball be thrown? (yes/no)"),
}


def features(df, cols):
    """model.features for its own columns; our extra columns as categories or floats."""
    base = [c for c in cols if c in m.ALL_FEATURES]
    X = m.features(df, base)
    for c in cols:
        if c in CATEGORICAL:
            X[c] = pd.Categorical(df[c].astype(object), categories=CATEGORIES[c])
        elif c not in base:
            X[c] = df[c].astype("float64")
    return X


def hgb():
    # Same settings as model._hgb (kept local so Modeler's edits there can't change these numbers).
    return HistGradientBoostingClassifier(early_stopping=True, categorical_features="from_dtype", random_state=0)


def evaluate(df, y, cols, group, first_season, test_season):
    keep = y.notna() & (df["season"] >= first_season)
    d, y = df[keep], y[keep].astype(str)
    train, test = split(d, test_season)
    ytr, yte = y.loc[train.index], y.loc[test.index]
    # A column with no values in training (e.g. box_part23 when testing on 2022) can't be binned; skip it.
    cols = [c for c in cols if train[c].notna().any()]
    model = hgb().fit(features(train, cols), ytr)
    classes = model.classes_
    p_model = model.predict_proba(features(test, cols))
    # Baseline: training class frequencies (per run/pass when the scheme mixes both; the model knows it too).
    if group:
        freq = pd.crosstab(train[group], ytr, normalize="index").reindex(columns=classes, fill_value=0)
        p_base = freq.loc[test[group]].to_numpy()
    else:
        p_base = np.tile(ytr.value_counts(normalize=True).reindex(classes, fill_value=0).to_numpy(), (len(test), 1))

    def score(p):
        return {"accuracy": float(np.mean(classes[p.argmax(1)] == yte.to_numpy())),
                "log_loss": float(log_loss(yte, p, labels=classes))}

    base, mod = score(p_base), score(p_model)
    # Paired per-play log-loss gap and its standard error: a skill smaller than ~2 SE is likely noise.
    idx = np.searchsorted(classes, yte.to_numpy())
    rows = np.arange(len(yte))
    gap = np.log(np.clip(p_model[rows, idx], 1e-15, 1)) - np.log(np.clip(p_base[rows, idx], 1e-15, 1))
    return {
        "n_train": len(train), "n_test": len(test),
        "test_balance": yte.value_counts(normalize=True).round(4).to_dict(),
        "baseline": base, "model": mod,
        "skill_accuracy": mod["accuracy"] - base["accuracy"],
        # Positive = model beats baseline. As a % it's comparable across schemes with different class counts.
        "skill_log_loss": base["log_loss"] - mod["log_loss"],
        "skill_log_loss_se": float(gap.std(ddof=1) / np.sqrt(len(gap))),
        "log_loss_reduction_pct": 100 * (1 - mod["log_loss"] / base["log_loss"]),
    }


def main():
    df = pd.read_parquet(DATA)
    CATEGORIES.update({c: sorted(df[c].dropna().astype(str).unique()) for c in CATEGORICAL})
    df = pd.concat([df, team_tendency(df)], axis=1)
    y_all = labels(df)

    results = {"setup": {
        "data": DATA, "test": "season S weeks 15-22; train = all earlier seasons + season S weeks 1-14",
        "baseline": "training class frequencies (per run/pass for D1/D5/D6); accuracy = always pick the top class",
        "skill": "model minus baseline (accuracy points; log loss drop, positive = better)",
        "D5_note": "FTN does not document whose view starting_hash is from. Reading A (same frame as "
                   "nflverse left/right) is used; under reading B hash_side and wide_side swap names, "
                   "every metric is identical.",
    }, "schemes": {}}
    for name, (cols, group, first, desc) in SCHEMES.items():
        y = y_all[name]
        r = {"description": desc,
             "balance_by_season": pd.crosstab(df["season"], y, normalize="index").round(4).T.to_dict()}
        for season in TEST_SEASONS:
            r[f"test_{season}"] = evaluate(df, y, cols, group, first, season)
            t = r[f"test_{season}"]
            print(f"{name:<11}{season}  n={t['n_test']:>6}  base acc {t['baseline']['accuracy']:.3f} "
                  f"ll {t['baseline']['log_loss']:.3f} | model acc {t['model']['accuracy']:.3f} "
                  f"ll {t['model']['log_loss']:.3f} | skill {t['skill_accuracy']:+.3f} "
                  f"ll -{t['log_loss_reduction_pct']:.1f}% "
                  f"(gap {t['skill_log_loss']:.4f} +/- {t['skill_log_loss_se']:.4f})")
        results["schemes"][name] = r

    # nflverse's short/deep comes from the play text, not air_yards >= 15, so check how often they disagree.
    thrown = df[(df["is_pass"] == 1) & df["pass_length"].notna() & df["air_yards"].notna()]
    disagree = (thrown["pass_length"] == "deep") != (thrown["air_yards"] >= 15)
    results["pass_length_vs_air_yards_15"] = {"throws": len(thrown), "disagree_share": float(disagree.mean())}
    dropbacks = df[df["is_pass"] == 1]
    results["no_throw_share_by_season"] = dropbacks.groupby("season")["air_yards"].apply(
        lambda a: round(float(a.isna().mean()), 4)).to_dict()
    print(f"pass_length vs air_yards>=15 disagree: {disagree.mean():.1%}")

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
