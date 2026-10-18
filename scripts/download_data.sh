#!/usr/bin/env bash
# Download raw data into data/raw/. Run from repo root:
#   bash scripts/download_data.sh          nflverse only (public, no login)
#   bash scripts/download_data.sh --bdb    also Big Data Bowl 2025 (needs Kaggle credentials +
#                                          accepted competition rules, see docs/PROGRESS.md)
set -euo pipefail

mkdir -p data/raw/nflverse
NFLVERSE=https://github.com/nflverse/nflverse-data/releases/download

# Fetch a file only if we don't already have it (each is ~3-20 MB).
get() {
  [ -f "$2" ] || curl -fL -o "$2" "$1"
}

# nflverse play-by-play + participation (formation, personnel, defenders in box), 2016-2025.
for Y in $(seq 2016 2025); do
  get "$NFLVERSE/pbp/play_by_play_$Y.parquet" "data/raw/nflverse/play_by_play_$Y.parquet"
  get "$NFLVERSE/pbp_participation/pbp_participation_$Y.parquet" "data/raw/nflverse/pbp_participation_$Y.parquet"
done

# FTN charting (hand-charted pre-snap details: hash, QB depth, motion, box). Only exists 2022+.
for Y in $(seq 2022 2025); do
  get "$NFLVERSE/ftn_charting/ftn_charting_$Y.parquet" "data/raw/nflverse/ftn_charting_$Y.parquet"
done

# Big Data Bowl 2025: tracking (weeks 1-9 of 2022) + plays/players/games.
if [ "${1:-}" = "--bdb" ]; then
  mkdir -p data/raw/bdb2025
  uv run kaggle competitions download -c nfl-big-data-bowl-2025 -p data/raw/bdb2025
  unzip -o -q data/raw/bdb2025/nfl-big-data-bowl-2025.zip -d data/raw/bdb2025
  rm data/raw/bdb2025/nfl-big-data-bowl-2025.zip
fi
