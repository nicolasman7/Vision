"""Convert a fitted sklearn HistGradientBoostingClassifier into a Core ML tree ensemble (.mlmodel).

coremltools 9.0 disables its own sklearn converter for sklearn 1.9, so we walk HGB's trees
ourselves and rebuild them with coremltools' TreeEnsembleClassifier builder.

How HGB predicts (what we copy):
    raw[k] = baseline[k] + sum of the leaf values each tree for class k lands in
    binary:     P(class 1) = sigmoid(raw)        multiclass: P = softmax(raw)
At each split: NaN goes left if missing_go_to_left, otherwise x <= threshold goes left.

Example: convert(hgb, ["down", "distance"], ["run", "pass"], calib=(0.9, -0.1)).save("run_pass.mlmodel")
"""
import json

import coremltools as ct
from coremltools.models import datatypes
from coremltools.models.tree_ensemble import TreeEnsembleClassifier


def convert(model, feature_names, class_labels, calib=None):
    """Return an MLModel: input `features` (double array, length F), outputs `label` and `probs`.

    class_labels: names in the same order as model.classes_ (e.g. ["run", "pass"]).
    calib: binary -> Platt (a, b), meaning P(class 1) = sigmoid(a * raw + b).
           multiclass -> temperature T, meaning P = softmax(raw / T).
           Both are linear in raw, so they fold exactly into the leaves and the base value.
    """
    n_classes = len(model.classes_)
    if len(class_labels) != n_classes:
        raise ValueError(f"{len(class_labels)} class_labels for a {n_classes}-class model")
    if len(feature_names) != model.n_features_in_:
        raise ValueError(f"{len(feature_names)} feature_names for a {model.n_features_in_}-feature model")

    binary = n_classes == 2
    baseline = model._baseline_prediction.ravel()  # (1,) binary, (n_classes,) multiclass
    # Leaves and base are both multiplied by `scale`; only the base gets `shift`.
    if binary:
        scale, shift = calib if calib is not None else (1.0, 0.0)
    else:
        scale, shift = (1.0 / calib if calib is not None else 1.0), 0.0

    builder = TreeEnsembleClassifier(
        [("features", datatypes.Array(len(feature_names)))], list(class_labels), ["label", "probs"])
    builder.set_default_prediction_value([float(scale * v + shift) for v in baseline])
    # Binary has one output dimension (the class-1 logit); Core ML fills P(class 0) = 1 - p.
    builder.set_post_evaluation_transform("Regression_Logistic" if binary else "Classification_SoftMax")

    tree_id = 0
    for iteration in model._predictors:
        for k, predictor in enumerate(iteration):  # one tree per class (just one for binary)
            nodes = predictor.nodes
            if nodes["is_categorical"].any():
                raise ValueError("categorical splits are not supported; one-hot or ordinal-encode those features")
            for node_id, n in enumerate(nodes):
                if n["is_leaf"]:
                    # Leaf values already include the learning rate.
                    builder.add_leaf_node(tree_id, node_id, {k: float(scale * n["value"])})
                else:
                    # x <= threshold is the true (left) child. Spike on macOS Core ML: with
                    # BranchOnValueLessThanEqual the missing flag is honored both ways, so NaN can
                    # be passed straight in. (With BranchOnValueGreaterThan NaN always went to the
                    # false child and the flag was ignored, so don't switch branch modes.)
                    builder.add_branch_node(
                        tree_id, node_id, int(n["feature_idx"]), float(n["num_threshold"]),
                        "BranchOnValueLessThanEqual", int(n["left"]), int(n["right"]),
                        missing_value_tracks_true_child=bool(n["missing_go_to_left"]))
            tree_id += 1

    mlmodel = ct.models.MLModel(builder.spec)
    # Feature order matters (the app must fill the array in this order), so ship it inside the model.
    mlmodel.user_defined_metadata["feature_names"] = json.dumps(list(feature_names))
    mlmodel.input_description["features"] = f"{len(feature_names)} doubles in feature_names order; NaN = missing"
    return mlmodel
