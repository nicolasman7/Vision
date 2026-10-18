import pandas as pd

from pipeline.prediction import model as m


def test_time_split_keeps_games_apart():
    df = pd.DataFrame({"game_id": ["g1", "g1", "g2", "g3"], "week": [3, 3, 14, 15]})
    train, test = m.time_split(df)
    assert train["week"].max() <= 14 < test["week"].min()
    assert not set(train["game_id"]) & set(test["game_id"])


def test_no_label_or_team_in_features():
    banned = {"direction", "air_yards", "posteam", "defteam", "game_id", "play_id", "week"}
    for cols in (m.SITUATION, m.ALL_FEATURES):
        assert not (set(m.LABELS) | banned) & set(cols)
    # is_pass is allowed only as an input to the direction model
    assert not banned & set(m.DIRECTION_FEATURES)
