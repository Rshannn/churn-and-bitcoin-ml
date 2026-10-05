"""The optional artifact must reproduce missing-value preprocessing locally."""
import math
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import train_telco


class ExportTests(unittest.TestCase):
    def test_artifact_carries_categorical_imputation_for_missing_input(self):
        frame = pd.DataFrame({
            "tenure": [1., 2., 3., 4., 5., 6., 7., 8.],
            "contract": pd.Series(["A", "A", "A", "B", np.nan, "B", "A", np.nan], dtype=object),
            "Churn": [0, 0, 1, 1, 0, 1, 0, 1],
        })
        preprocessor, x, y, numeric, categorical = train_telco.build_preprocessor(frame)
        pipe = Pipeline([("preprocessor", preprocessor), ("clf", SGDClassifier(loss="log_loss", random_state=42))])
        pipe.fit(x, y)
        artifact = train_telco.build_lightweight_artifact(frame, preprocessor, numeric, categorical, pipe)
        self.assertEqual(artifact.get("categorical_imputer_modes"), ["A"])
        query = pd.DataFrame({"tenure": [np.nan], "contract": pd.Series([np.nan], dtype=object)})
        expected = pipe.predict_proba(query)[0, 1]
        num = (artifact["numeric_imputer_medians"][0] - artifact["numeric_means"][0]) / artifact["numeric_scales"][0]
        category = artifact["categorical_imputer_modes"][0]
        encoded = [num] + [float(value == category) for value in artifact["categories"][0]]
        score = sum(w * value for w, value in zip(artifact["coef"][0], encoded)) + artifact["intercept"][0]
        reconstructed = 1. / (1. + math.exp(-score))
        self.assertAlmostEqual(reconstructed, expected, places=10)


if __name__ == "__main__":
    unittest.main()
