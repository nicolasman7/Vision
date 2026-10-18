import pandas as pd
import pytest

from pipeline.tracking.track import COLUMNS, clip_stats


def test_clip_stats_known_answers():
    # 10 frames. ID 1 (person) in frames 0-9, ID 2 (person) in frames 0-3, ID 3 (ball) in frames 2 and 5.
    rows = [(f, 1, 0, 0.9, 0, 0, 1, 1) for f in range(10)]
    rows += [(f, 2, 0, 0.8, 0, 0, 1, 1) for f in range(4)]
    rows += [(f, 3, 32, 0.3, 0, 0, 1, 1) for f in (2, 5)]
    s = clip_stats(pd.DataFrame(rows, columns=COLUMNS), n_frames=10, fps=10)

    assert s["persons_mean"] == pytest.approx(1.4)  # (4*2 + 6*1) / 10
    assert s["persons_median"] == 1.0
    assert s["ball_frame_rate"] == pytest.approx(0.2)
    assert s["n_ids"] == 3
    assert s["median_track_len"] == 4.0  # lengths 10, 4, 2
    assert s["ids_per_100_frames"] == pytest.approx(30.0)
    assert s["seconds"] == 1.0


def test_clip_stats_counts_empty_frames_as_zero():
    df = pd.DataFrame([(0, 1, 0, 0.9, 0, 0, 1, 1)], columns=COLUMNS)
    s = clip_stats(df, n_frames=4, fps=30)
    assert s["persons_mean"] == 0.25 and s["persons_median"] == 0.0
