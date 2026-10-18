#!/usr/bin/env bash
# Download raw data into data/raw/. Run from repo root: bash scripts/download_data.sh
# BDB 2025 needs Kaggle credentials + accepted competition rules (see docs/PROGRESS.md).
set -euo pipefail

mkdir -p data/raw/nflverse data/raw/bdb2025

# nflverse play-by-play, 2022 season (same season as BDB 2025). Public, no login.
curl -fL -o data/raw/nflverse/play_by_play_2022.parquet \
  https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_2022.parquet

# nflverse participation, 2022: formation, personnel, defenders in box. Public, no login.
curl -fL -o data/raw/nflverse/pbp_participation_2022.parquet \
  https://github.com/nflverse/nflverse-data/releases/download/pbp_participation/pbp_participation_2022.parquet

# Big Data Bowl 2025: tracking (weeks 1-9 of 2022) + plays/players/games.
uv run kaggle competitions download -c nfl-big-data-bowl-2025 -p data/raw/bdb2025
unzip -o -q data/raw/bdb2025/nfl-big-data-bowl-2025.zip -d data/raw/bdb2025
rm data/raw/bdb2025/nfl-big-data-bowl-2025.zip
