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


def test_season_split_never_trains_on_later_weeks():
    df = pd.DataFrame([(s, w) for s in (2020, 2021, 2022) for w in (1, 14, 15, 18)], columns=["season", "week"])
    train, test = m.season_split(df, 2021, last_train_week=14)
    assert set(zip(test.season, test.week)) == {(2021, 15), (2021, 18)}
    assert all((s, w) < (2021, 15) for s, w in zip(train.season, train.week))
    train, _ = m.season_split(df, 2021, last_train_week=14, train_seasons=[2021])
    assert set(train.season) == {2021}


def test_corruption_changes_requested_fraction():
    n = 20_000
    X = m.features(pd.DataFrame({"offense_formation": ["SHOTGUN"] * n, "defenders_in_box": [6.0] * n}),
                   ["offense_formation", "defenders_in_box"])
    bad = m.corrupt_category(X, "offense_formation", 0.2, ["SHOTGUN", "SINGLEBACK", "EMPTY"])
    assert abs((bad["offense_formation"] != "SHOTGUN").mean() - 0.2) < 0.02
    assert set(bad["offense_formation"].cat.categories) == set(m.FORMATIONS)
    shifted = m.shift_count(X, "defenders_in_box", 0.3)
    assert abs((shifted["defenders_in_box"] != 6).mean() - 0.3) < 0.02
    assert set(shifted["defenders_in_box"]) == {5, 6, 7}
