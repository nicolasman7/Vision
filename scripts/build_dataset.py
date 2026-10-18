"""Build data/processed/plays_<year>.parquet for 2016-2025 plus plays_2016_2025.parquet combined.

Run from repo root: uv run python -m scripts.build_dataset
One season at a time, reading only the play-by-play columns we need, so it fits in 16 GB.
"""

import os

import pandas as pd

from pipeline.prediction.dataset import PBP_COLUMNS, build_dataset

RAW = "data/raw/nflverse"
OUT = "data/processed"
SEASONS = range(2016, 2026)

summary = []
for year in SEASONS:
    print(f"\n=== {year} ===")
    pbp = pd.read_parquet(f"{RAW}/play_by_play_{year}.parquet", columns=PBP_COLUMNS)
    part = pd.read_parquet(f"{RAW}/pbp_participation_{year}.parquet")
    ftn_path = f"{RAW}/ftn_charting_{year}.parquet"
    ftn = pd.read_parquet(ftn_path) if os.path.exists(ftn_path) else None  # FTN exists 2022+
    df = build_dataset(pbp, part, ftn, verbose=True)
    df.to_parquet(f"{OUT}/plays_{year}.parquet", index=False)
    summary.append({"season": year, "plays": len(df), "pass_rate": df["is_pass"].mean(),
                    "formation_family_fill": df["formation_family"].notna().mean(),
                    "ftn_fill": df["ftn_qb_location"].notna().mean()})

all_df = pd.concat([pd.read_parquet(f"{OUT}/plays_{y}.parquet") for y in SEASONS], ignore_index=True)
all_df.to_parquet(f"{OUT}/plays_2016_2025.parquet", index=False)

pd.set_option("display.width", 200)
print("\nPER SEASON\n", pd.DataFrame(summary).round(3).to_string(index=False))
print(f"\nwrote {OUT}/plays_2016_2025.parquet: {all_df.shape}")
print("\nnulls by season (share):\n", all_df.isna().groupby(all_df["season"]).mean().T.round(2).to_string())
print("\nsample:\n", all_df.sample(3, random_state=0).T.to_string())
