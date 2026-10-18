"""Parity: a converted HGB must give the same probabilities through Core ML as in Python, NaN rows included."""
import coremltools as ct
import numpy as np
import pytest
from scipy.special import expit, softmax
from sklearn.ensemble import HistGradientBoostingClassifier

from pipeline.export.hgb_to_coreml import convert

N_FEATURES = 5


def _data(n_classes, n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, N_FEATURES))
    y = (X[:, 0] + X[:, 1] > 0).astype(int) + (n_classes == 3) * (X[:, 2] > 0.5)
    X[rng.random(X.shape) < 0.15] = np.nan  # scattered missing values in every column
    X[:20, :] = np.nan  # some rows fully missing
    return X, y


@pytest.mark.parametrize("n_classes, calib", [(2, None), (2, (0.8, -0.3)), (3, 1.7)])
def test_coreml_matches_python(tmp_path, n_classes, calib):
    X, y = _data(n_classes)
    hgb = HistGradientBoostingClassifier(max_iter=30, random_state=0).fit(X, y)
    labels = ["run", "pass"] if n_classes == 2 else ["left", "middle", "right"]

    path = tmp_path / "m.mlmodel"
    convert(hgb, [f"f{i}" for i in range(N_FEATURES)], labels, calib).save(str(path))
    mlmodel = ct.models.MLModel(str(path))

    X_test, _ = _data(n_classes, n=200, seed=1)
    raw = hgb.decision_function(X_test)
    if n_classes == 2:
        a, b = calib or (1.0, 0.0)
        p1 = expit(a * raw + b)
        expected = np.column_stack([1 - p1, p1])
    else:
        expected = softmax(raw / calib, axis=1)

    out = mlmodel.predict([{"features": row} for row in X_test])
    got = np.array([[o["probs"][c] for c in labels] for o in out])

    assert np.abs(got - expected).max() <= 1e-4
    # Argmax must agree unless the top two Python probabilities are within 1e-4 (a tie).
    top2 = np.sort(expected, axis=1)[:, -2:]
    clear = top2[:, 1] - top2[:, 0] > 1e-4
    assert (got.argmax(1) == expected.argmax(1))[clear].all()
    assert all(o["label"] == labels[i] for o, i in zip(out, got.argmax(1)))


def test_categorical_splits_rejected():
    X, y = _data(2)
    X[:, 0] = 3 * y  # a perfectly predictive category, so the first tree splits on it
    hgb = HistGradientBoostingClassifier(max_iter=5, categorical_features=[0]).fit(X, y)
    with pytest.raises(ValueError, match="categorical"):
        convert(hgb, [f"f{i}" for i in range(N_FEATURES)], ["run", "pass"])
