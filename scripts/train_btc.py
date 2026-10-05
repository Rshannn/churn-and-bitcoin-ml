# ============================================================
# DSCI 352 Final Project
# Bitcoin high-price forecasting with LSTM
# ============================================================

import os
import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"))
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# -----------------------------
# Configuration
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_FILE = PROJECT_ROOT / "data" / "btc_price.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "btc"

LOOKBACK = 30
EPOCHS = 50
BATCH_SIZE = 32

TRAIN_START = "2019-01-01"
TRAIN_END = "2022-12-31"
TEST_START = "2023-01-01"
TEST_END = "2025-12-31"


# -----------------------------
# Helper functions
# -----------------------------

def create_windows(data, lookback=30):
    """
    Convert a time series into sliding-window samples.

    Example:
    previous 30 observations -> next observation
    """
    if lookback < 1 or len(data) <= lookback:
        raise ValueError(f"Need more than {lookback} observations and a positive lookback.")
    X = []
    y = []

    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i, 0])
        y.append(data[i, 0])

    X = np.array(X)
    y = np.array(y)

    X = X.reshape((X.shape[0], X.shape[1], 1))

    return X, y


def build_lstm_model(lookback):
    """
    Build and compile an LSTM regression model.
    """
    model = Sequential()

    model.add(tf.keras.Input(shape=(lookback, 1)))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(32))
    model.add(Dropout(0.2))

    model.add(Dense(1))

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    return model


# -----------------------------
# Main script
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Train the original Bitcoin high-price LSTM and report a persistence baseline.")
    parser.add_argument("--input", type=Path, default=CSV_FILE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    tf.keras.utils.set_random_seed(42)

    print("Loading BTC data...")

    df = pd.read_csv(args.input)

    print("Original columns:")
    print(df.columns.tolist())

    if not {"Date", "High"}.issubset(df.columns):
        raise ValueError("Input CSV must contain Date and High columns.")
    # Keep only required columns
    df = df[["Date", "High"]].copy()

    # Clean date and price columns
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", format="mixed")
    df["High"] = pd.to_numeric(df["High"], errors="coerce")

    # Drop missing rows
    input_rows = len(df)
    df = df.dropna(subset=["Date", "High"])
    dropped_rows = input_rows - len(df)
    if df["Date"].duplicated().any():
        raise ValueError("Duplicate dates are not supported.")
    if not np.isfinite(df["High"]).all():
        raise ValueError("High prices must be finite.")

    # Sort oldest to newest
    df = df.sort_values("Date").reset_index(drop=True)

    print("\nCleaned data preview:")
    print(df.head())
    print(df.tail())

    print("\nDate range:")
    print("Start:", df["Date"].min())
    print("End:", df["Date"].max())
    print("Rows:", df.shape[0])

    # -----------------------------
    # Plot original BTC High price
    # -----------------------------

    plt.figure(figsize=(12, 5))
    plt.plot(df["Date"], df["High"])
    plt.title("Bitcoin Daily High Price")
    plt.xlabel("Date")
    plt.ylabel("High Price")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "btc_high_price_series.png"), dpi=300)
    plt.close()

    print(f"Saved plot: {output_dir / 'btc_high_price_series.png'}")

    # -----------------------------
    # Train-test split
    # -----------------------------

    train_df = df[(df["Date"] >= TRAIN_START) & (df["Date"] <= TRAIN_END)].copy()
    test_df = df[(df["Date"] >= TEST_START) & (df["Date"] <= TEST_END)].copy()

    print("\nTrain shape:", train_df.shape)
    print("Test shape:", test_df.shape)

    if train_df.empty:
        raise ValueError("Training set is empty. Check your date range.")
    if test_df.empty:
        raise ValueError("Test set is empty. Check your date range.")

    train_values = train_df[["High"]].values
    test_values = test_df[["High"]].values

    # -----------------------------
    # Scale data
    # -----------------------------

    scaler = MinMaxScaler(feature_range=(0, 1))

    train_scaled = scaler.fit_transform(train_values)
    test_scaled = scaler.transform(test_values)

    # -----------------------------
    # Create sliding windows
    # -----------------------------

    X_train, y_train = create_windows(train_scaled, LOOKBACK)
    X_test, y_test = create_windows(test_scaled, LOOKBACK)

    print("\nWindowed data shapes:")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)

    # -----------------------------
    # Build and train model
    # -----------------------------

    print("\nBuilding LSTM model...")

    model = build_lstm_model(LOOKBACK)
    model.summary()

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    print("\nTraining model...")

    history = model.fit(
        X_train,
        y_train,
        epochs=args.epochs,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )

    # -----------------------------
    # Plot training loss
    # -----------------------------

    plt.figure(figsize=(10, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("LSTM Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lstm_training_loss.png"), dpi=300)
    plt.close()

    print(f"Saved plot: {output_dir / 'lstm_training_loss.png'}")

    # -----------------------------
    # Predict test set
    # -----------------------------

    print("\nPredicting test prices...")

    pred_scaled = model.predict(X_test)

    pred_prices = scaler.inverse_transform(pred_scaled)
    actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

    test_dates = test_df["Date"].iloc[LOOKBACK:].reset_index(drop=True)

    # -----------------------------
    # Evaluate
    # -----------------------------

    rmse = np.sqrt(mean_squared_error(actual_prices, pred_prices))
    mae = mean_absolute_error(actual_prices, pred_prices)

    print("\nEvaluation Results")
    print("------------------")
    print(f"Test RMSE: {rmse:.2f}")
    print(f"Test MAE: {mae:.2f}")

    # Save metrics to text file
    metrics_path = os.path.join(output_dir, "btc_lstm_metrics.txt")

    with open(metrics_path, "w") as f:
        f.write("Bitcoin LSTM Forecasting Results\n")
        f.write("--------------------------------\n")
        f.write(f"Lookback window: {LOOKBACK} days\n")
        f.write(f"Training period: {TRAIN_START} to {TRAIN_END}\n")
        f.write(f"Testing period: {TEST_START} to {TEST_END}\n")
        f.write(f"Test RMSE: {rmse:.2f}\n")
        f.write(f"Test MAE: {mae:.2f}\n")

    print(f"\nSaved metrics: {metrics_path}")

    # -----------------------------
    # Plot actual vs predicted
    # -----------------------------

    plt.figure(figsize=(12, 5))
    plt.plot(test_dates, actual_prices, label="Actual High Price")
    plt.plot(test_dates, pred_prices, label="Predicted High Price")
    plt.title("Bitcoin High Price Forecast: Actual vs Predicted")
    plt.xlabel("Date")
    plt.ylabel("High Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "btc_actual_vs_predicted.png"), dpi=300)
    plt.close()

    print(f"Saved plot: {output_dir / 'btc_actual_vs_predicted.png'}")

    # -----------------------------
    # Save model
    # -----------------------------

    model_path = args.model_dir / "btc_lstm_model.keras"
    model.save(model_path)

    scaler_path = args.model_dir / "btc_scaler.pkl"
    joblib.dump(scaler, scaler_path)

    print(f"Saved model: {model_path}")
    print(f"Saved scaler: {scaler_path}")

    # Persistence uses the last observed High in exactly the same test windows.
    baseline_prices = test_values[LOOKBACK - 1:-1, 0]
    actual = actual_prices.ravel()
    predicted = pred_prices.ravel().astype("float64")
    baseline_rmse = float(np.sqrt(mean_squared_error(actual, baseline_prices)))
    baseline_mae = float(mean_absolute_error(actual, baseline_prices))
    predictions = pd.DataFrame({
        "Date": test_dates,
        "Actual_High": actual,
        "Predicted_High": predicted,
        "Persistence_High": baseline_prices,
        "Absolute_Error": np.abs(actual - predicted),
        "Persistence_Absolute_Error": np.abs(actual - baseline_prices),
    })
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    comparison = pd.DataFrame([
        {"model": "lstm", "rmse": float(rmse), "mae": float(mae)},
        {"model": "previous_observation", "rmse": baseline_rmse, "mae": baseline_mae},
    ])
    comparison.to_csv(output_dir / "model_comparison_btc.csv", index=False)
    date_string = lambda value: value.strftime("%Y-%m-%d")
    metadata = {
        "seed": 42,
        "lookback_observations": LOOKBACK,
        "epochs_requested": args.epochs,
        "epochs_run": len(history.history["loss"]),
        "input_rows": input_rows,
        "dropped_rows": dropped_rows,
        "data_start": date_string(df["Date"].min()),
        "data_end": date_string(df["Date"].max()),
        "non_daily_intervals": int((df["Date"].diff().dropna() != pd.Timedelta(days=1)).sum()),
        "train_start": date_string(train_df["Date"].min()),
        "train_end": date_string(train_df["Date"].max()),
        "test_input_start": date_string(test_df["Date"].min()),
        "test_input_end": date_string(test_df["Date"].max()),
        "prediction_start": date_string(test_dates.min()),
        "prediction_end": date_string(test_dates.max()),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "lstm_rmse": float(rmse),
        "lstm_mae": float(mae),
        "baseline_rmse": baseline_rmse,
        "baseline_mae": baseline_mae,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with open(metrics_path, "w") as f:
        f.write("Bitcoin high-price rolling evaluation\n")
        f.write(f"Lookback: {LOOKBACK} previous observations\n")
        f.write(f"Actual training data: {metadata['train_start']} to {metadata['train_end']}\n")
        f.write(f"Evaluated predictions: {metadata['prediction_start']} to {metadata['prediction_end']}\n")
        f.write(f"Test observations: {len(X_test)}\n")
        f.write(f"LSTM RMSE: {rmse:.2f}; MAE: {mae:.2f}\n")
        f.write(f"Persistence RMSE: {baseline_rmse:.2f}; MAE: {baseline_mae:.2f}\n")
    print("\nSame-window model comparison:")
    print(comparison.to_string(index=False))



if __name__ == "__main__":
    main()