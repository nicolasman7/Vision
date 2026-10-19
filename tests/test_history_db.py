import os
import sqlite3

import pandas as pd
import pytest

DB = "coreml/history.sqlite"


@pytest.mark.skipif(not os.path.exists(DB), reason="run: uv run python -m scripts.build_history_db")
def test_history_db_matches_parquet():
    con = sqlite3.connect(DB)
    expected = sum(len(pd.read_parquet(f"data/processed/plays_{y}.parquet", columns=["play_id"]))
                   for y in range(2016, 2026))
    assert con.execute("SELECT COUNT(*) FROM plays").fetchone()[0] == expected
    # in_training = 1 exactly for seasons <= 2023 (season is the first 4 chars of game_id)
    bad = con.execute("SELECT COUNT(*) FROM plays WHERE in_training != (CAST(substr(game_id, 1, 4) AS INT) <= 2023)")
    assert bad.fetchone()[0] == 0
