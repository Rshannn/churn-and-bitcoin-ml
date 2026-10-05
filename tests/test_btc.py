"""Regression checks for input windows and portable prediction defaults."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import train_btc
import predict_btc


class WindowTests(unittest.TestCase):
    def test_training_windows_use_only_previous_observations(self):
        x, y = train_btc.create_windows(np.arange(5).reshape(-1, 1), 3)
        np.testing.assert_array_equal(x[:, :, 0], [[0, 1, 2], [1, 2, 3]])
        np.testing.assert_array_equal(y, [3, 4])

    def test_prediction_windows_match_training_inputs(self):
        x, indices = predict_btc.create_windows(np.arange(5).reshape(-1, 1), 3)
        np.testing.assert_array_equal(x[:, :, 0], [[0, 1, 2], [1, 2, 3]])
        self.assertEqual(indices, [3, 4])

    def test_training_rejects_insufficient_history(self):
        caught = None
        try:
            train_btc.create_windows(np.arange(3).reshape(-1, 1), 3)
        except Exception as exc:
            caught = exc
        self.assertIsInstance(caught, ValueError)

    def test_prediction_rejects_insufficient_history(self):
        caught = None
        try:
            predict_btc.create_windows(np.arange(3).reshape(-1, 1), 3)
        except Exception as exc:
            caught = exc
        self.assertIsInstance(caught, ValueError)

    def test_default_training_data_loads_outside_project_directory(self):
        import os
        import tempfile
        import pandas as pd
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                caught = None
                try:
                    data = pd.read_csv(train_btc.CSV_FILE)
                except Exception as exc:
                    caught = exc
                self.assertIsNone(caught, str(caught))
                self.assertGreater(len(data), 30)
            finally:
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
