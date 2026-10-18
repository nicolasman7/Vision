"""Build data/processed/plays_2022.parquet. Run from repo root: uv run python -m scripts.build_dataset"""

import pandas as pd

from pipeline.prediction.dataset import build_dataset

RAW = "data/raw/nflverse"
OUT = "data/processed/plays_2022.parquet"

pbp = pd.read_parquet(f"{RAW}/play_by_play_2022.parquet")
part = pd.read_parquet(f"{RAW}/pbp_participation_2022.parquet")
df = build_dataset(pbp, part, verbose=True)
df.to_parquet(OUT, index=False)

pd.set_option("display.width", 200)
print(f"\nwrote {OUT}: {df.shape}")
print("\nis_pass:\n", df["is_pass"].value_counts(normalize=True).round(3).to_string())
print("\ndirection:\n", df["direction"].value_counts(dropna=False).to_string())
print("\nnulls:\n", df.isna().sum()[lambda s: s > 0].to_string())
print("\nsample:\n", df.sample(3, random_state=0).T.to_string())
