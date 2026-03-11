import os
import numpy as np

from tabpfnwide.classifier import TabPFNWideClassifier


if __name__ == "__main__":
    # Small, linearly separable toy dataset
    X = np.array(
        [
            [0.0, 0.0, 0.0, 0.1, 0.2, 0.3],
            [0.1, 0.0, 0.0, 0.2, 0.1, 0.4],
            [1.0, 1.0, 1.0, 1.2, 1.1, 1.3],
            [1.1, 0.9, 1.0, 1.3, 1.2, 1.4],
            [0.0, 0.2, 0.1, 0.0, 0.2, 0.1],
            [0.9, 1.1, 1.0, 1.1, 1.2, 1.0],
            [0.2, 0.1, 0.2, 0.1, 0.3, 0.2],
            [1.2, 1.0, 1.1, 1.0, 1.1, 1.2],
        ],
        dtype=np.float32,
    )
    y = np.array([0, 0, 1, 1, 0, 1, 0, 1], dtype=np.int64)

    clf = TabPFNWideClassifier(
        model_name="wide-v2-5k",
        n_estimators=1,
        features_per_group=1,
        device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
    )

    clf.fit(X, y)
    probas = clf.predict_proba(X)
    preds = clf.predict(X)

    print("Predicted probabilities:\n", probas)
    print("Predicted labels:\n", preds)
    if np.array_equal(preds, y):
        print("MATCH: Predictions match the true labels.")
    else:
        print("NOT MATCH: Predictions do not match the true labels.")
