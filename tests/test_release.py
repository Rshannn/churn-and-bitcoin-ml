"""Artifact integrity and CLI checks for the verified release."""
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


class ReleaseChecks(unittest.TestCase):
    def test_baseline_uses_previous_observation_on_same_test_dates(self):
        rows = read_rows(ROOT / "results/btc/test_predictions.csv")
        raw = read_rows(ROOT / "data/btc_price.csv")
        from datetime import datetime
        values = sorted((datetime.strptime(r["Date"], "%m/%d/%Y").strftime("%Y-%m-%d"), float(r["High"])) for r in raw)
        previous = {date: values[i - 1][1] for i, (date, value) in enumerate(values) if i}
        self.assertEqual(len(rows), 296)
        self.assertEqual(rows[0]["Date"][:10], "2023-01-31")
        self.assertEqual(rows[-1]["Date"][:10], "2023-11-22")
        for row in rows:
            self.assertAlmostEqual(float(row["Persistence_High"]), previous[row["Date"][:10]], places=6)
        errors = [float(r["Actual_High"]) - float(r["Persistence_High"]) for r in rows]
        self.assertAlmostEqual(math.sqrt(sum(e * e for e in errors) / len(errors)), 668.3206041507567, places=6)

    def test_reported_metrics_match_saved_predictions(self):
        rows = read_rows(ROOT / "results/btc/test_predictions.csv")
        metadata = json.loads((ROOT / "results/btc/run_metadata.json").read_text())
        errors = [float(r["Actual_High"]) - float(r["Predicted_High"]) for r in rows]
        self.assertAlmostEqual(math.sqrt(sum(e * e for e in errors) / len(errors)), metadata["lstm_rmse"], places=6)
        self.assertAlmostEqual(sum(abs(e) for e in errors) / len(errors), metadata["lstm_mae"], places=6)

    def test_saved_model_cli_matches_training_predictions(self):
        training = read_rows(ROOT / "results/btc/test_predictions.csv")
        inference = {r["Date"][:10]: r for r in read_rows(ROOT / "results/btc/btc_cli_predictions.csv")}
        for row in training:
            predicted = float(inference[row["Date"][:10]]["Predicted_High"])
            self.assertAlmostEqual(predicted, float(row["Predicted_High"]), delta=0.05)

    def run_invalid_input(self, text):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "input.csv").write_text(text)
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts/predict_btc.py"), "--input", str(path / "input.csv"), "--output", str(path / "output.csv")],
                cwd=directory, capture_output=True, text=True,
                env=dict(os.environ, TF_CPP_MIN_LOG_LEVEL="3", TF_NUM_INTRAOP_THREADS="2", TF_NUM_INTEROP_THREADS="2"),
                timeout=90,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((path / "output.csv").exists())
            return result.stderr

    def test_cli_rejects_short_input_before_model_inference(self):
        from datetime import date, timedelta
        rows = [f"{date(2023, 1, 1) + timedelta(days=i)},{i+100}" for i in range(30)]
        error = self.run_invalid_input("Date,High\n" + "\n".join(rows) + "\n")
        self.assertIn("more than 30 rows", error)

    def test_cli_rejects_duplicate_dates(self):
        error = self.run_invalid_input("Date,High\n2023-01-01,100\n2023-01-01,101\n")
        self.assertIn("Duplicate dates", error)

    def test_cli_rejects_nonfinite_prices(self):
        error = self.run_invalid_input("Date,High\n2023-01-01,inf\n")
        self.assertIn("finite", error)


if __name__ == "__main__":
    unittest.main()
