# ============================================================
# DSCI 352 Final Project
# Rolling historical CLI prediction for the BTC LSTM
# ============================================================

import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from tensorflow.keras.models import load_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOOKBACK = 30


def create_windows(data, lookback=30):
    """
    Convert scaled BTC high prices into LSTM input windows.
    """
    if lookback < 1 or len(data) <= lookback:
        raise ValueError(f"Need more than {lookback} observations and a positive lookback.")
    X = []
    dates = []

    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i, 0])
        dates.append(i)

    X = np.array(X)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    return X, dates


def main():
    parser = argparse.ArgumentParser(
        description="Predict Bitcoin high prices using a trained LSTM model."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to input CSV file containing Date and High columns."
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path to output CSV file where predictions will be saved."
    )

    parser.add_argument(
        "--model",
        default=str(PROJECT_ROOT / "models" / "btc_lstm_model.keras"),
        help="Path to trained Keras LSTM model."
    )

    parser.add_argument(
        "--scaler",
        default=str(PROJECT_ROOT / "models" / "btc_scaler.pkl"),
        help="Path to saved MinMaxScaler."
    )

    args = parser.parse_args()

    print("Loading input data...")
    df = pd.read_csv(args.input)

    if "Date" not in df.columns or "High" not in df.columns:
        raise ValueError("Input CSV must contain 'Date' and 'High' columns.")

    df = df[["Date", "High"]].copy()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", format="mixed")
    df["High"] = pd.to_numeric(df["High"], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["Date", "High"])
    if len(df) != before:
        print(f"Dropped {before - len(df)} rows with invalid dates or High prices.")
    if df["Date"].duplicated().any():
        raise ValueError("Duplicate dates are not supported.")
    if not np.isfinite(df["High"]).all():
        raise ValueError("High prices must be finite.")
    df = df.sort_values("Date").reset_index(drop=True)

    if len(df) <= LOOKBACK:
        raise ValueError(
            f"Input file must contain more than {LOOKBACK} rows for rolling prediction."
        )

    print("Loading model and scaler...")
    model = load_model(args.model, compile=False)
    if model.input_shape[1] != LOOKBACK:
        raise ValueError(f"Model must accept {LOOKBACK}-observation windows.")
    scaler = joblib.load(args.scaler)

    high_values = df[["High"]].values
    high_scaled = scaler.transform(high_values)

    X, _ = create_windows(high_scaled, LOOKBACK)

    print("Generating predictions...")
    pred_scaled = model.predict(X)
    pred_prices = scaler.inverse_transform(pred_scaled)

    prediction_dates = df["Date"].iloc[LOOKBACK:].reset_index(drop=True)
    actual_prices = df["High"].iloc[LOOKBACK:].reset_index(drop=True)

    output_df = pd.DataFrame({
        "Date": prediction_dates,
        "Actual_High": actual_prices,
        "Predicted_High": pred_prices.flatten().astype("float64")
    })

    output_df["Absolute_Error"] = (
        output_df["Actual_High"] - output_df["Predicted_High"]
    ).abs()

    os.makedirs(os.path.dirname(args.output), exist_ok=True) if os.path.dirname(args.output) else None

    output_df.to_csv(args.output, index=False)

    print(f"Predictions saved to: {args.output}")
    print(output_df.head())


if __name__ == "__main__":
    main()