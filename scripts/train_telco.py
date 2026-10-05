"""Telco churn model comparison. Adapted from a completed DSCI 352 course project."""

import os
import json
import argparse
from pathlib import Path

# Keep optional plotting imports from trying to write under the home directory.
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_CSV = PROJECT_ROOT / "data" / "telco_churn.csv"

# Optional AWS export configuration
S3_BUCKET = os.environ.get("MODEL_BUCKET", "")
S3_KEY_LIGHT = os.environ.get("MODEL_KEY", "models/telco_churn_light.json")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")


# Data loading and preprocessing
def load_telco(path: str) -> pd.DataFrame:
    """Load the Telco data, clean TotalCharges, encode Churn, and drop customerID."""
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    df = df.drop(columns=["customerID"], errors="ignore")
    return df


def build_preprocessor(df: pd.DataFrame):
    """Build numeric and categorical preprocessing and return features and labels."""
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = [
        col
        for col in X.columns
        if (
            pd.api.types.is_object_dtype(X[col])
            or pd.api.types.is_string_dtype(X[col])
            or pd.api.types.is_bool_dtype(X[col])
        )
    ]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor, X, y, numeric_features, categorical_features


def make_numpy_for_keras(preprocessor, X_train, X_val):
    """Fit preprocessing on training rows only and transform both splits."""
    preprocessor.fit(X_train)
    X_train_proc = preprocessor.transform(X_train)
    X_val_proc = preprocessor.transform(X_val)

    ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_names = ohe.get_feature_names_out()
    num_names = preprocessor.transformers_[0][2]
    feature_names = np.concatenate([num_names, cat_names])

    return X_train_proc.astype("float32"), X_val_proc.astype("float32"), feature_names


class GradientTracker(keras.callbacks.Callback):
    """Record gradient statistics on a fixed training batch after each epoch."""

    def __init__(self, model, sample_x, sample_y):
        super().__init__()
        self.model_for_grads = model
        self.sample_x = tf.convert_to_tensor(sample_x)
        self.sample_y = tf.convert_to_tensor(sample_y, dtype=tf.float32)
        self.history = []

    def on_epoch_end(self, epoch, logs=None):
        with tf.GradientTape() as tape:
            preds = self.model_for_grads(self.sample_x, training=True)  # (batch, 1)
            preds = tf.squeeze(preds, axis=-1)  # -> (batch,)
            loss = keras.losses.binary_crossentropy(self.sample_y, preds)
            loss = tf.reduce_mean(loss)

        grads = tape.gradient(loss, self.model_for_grads.trainable_weights)
        grad_vals = [tf.reshape(g, [-1]) for g in grads if g is not None]
        if grad_vals:
            all_grads = tf.concat(grad_vals, axis=0)
            grad_mean = tf.reduce_mean(tf.abs(all_grads)).numpy().item()
            grad_max = tf.reduce_max(tf.abs(all_grads)).numpy().item()
            grad_std = tf.math.reduce_std(all_grads).numpy().item()
        else:
            grad_mean = grad_max = grad_std = 0.0

        self.history.append(
            {
                "epoch": int(epoch),
                "grad_abs_mean": grad_mean,
                "grad_abs_max": grad_max,
                "grad_std": grad_std,
                "loss": float(loss.numpy()),
                "logs": logs or {},
            }
        )


# ##################
# Keras Model
def build_keras_model(input_dim: int):
    """Build the original dense neural network for binary churn prediction."""
    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(64, activation="relu", kernel_initializer="he_normal"),
            layers.Dropout(0.25),
            layers.Dense(32, activation="relu", kernel_initializer="he_normal"),
            layers.Dropout(0.15),
            layers.Dense(16, activation="relu", kernel_initializer="he_normal"),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="telco_churn_mlp",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc"), "accuracy"],
    )

    return model

############################
# Scikit-Learn Models
def train_sklearn_models(preprocessor, X, y):
    """Train four classifiers on one stratified split and rank validation AUC."""
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model_specs = [
        (
            "logistic_regression",
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        ),
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        ),
        (
            "gradient_boosting",
            GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=3,
                random_state=42,
            ),
        ),
        (
            "sgd_log_loss",
            SGDClassifier(
                loss="log_loss",
                penalty="elasticnet",
                alpha=1e-4,
                l1_ratio=0.15,
                max_iter=2000,
                tol=1e-4,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ]

    results = []
    sgd_model = None
    sgd_auc = None

    for name, clf in model_specs:
        pipe = Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                ("clf", clf),
            ]
        )
        pipe.fit(X_train, y_train)

        if hasattr(pipe.named_steps["clf"], "predict_proba"):
            proba = pipe.predict_proba(X_val)[:, 1]
        else:
            scores = pipe.decision_function(X_val)
            proba = 1.0 / (1.0 + np.exp(-scores))

        preds = (proba >= 0.5).astype(int)
        auc_val = roc_auc_score(y_val, proba)
        acc_val = accuracy_score(y_val, preds)
        results.append((name, pipe, auc_val, acc_val))

        print(f"  {name:20s} AUC={auc_val:.4f} ACC={acc_val:.4f}")

        if name == "sgd_log_loss":
            sgd_model = pipe
            sgd_auc = auc_val

    results.sort(key=lambda row: row[2], reverse=True)

    return results, (X_train, X_val, y_train, y_val), sgd_model, sgd_auc


################################################
# 4) Lightweight Artifact for AWS (Bonus)
def build_lightweight_artifact(
    df,
    preprocessor,
    numeric_features,
    categorical_features,
    sgd_model,
):
    """Refit an SGD pipeline on all rows and export compatible preprocessing and coefficients."""
    X_full = df.drop("Churn", axis=1)
    y_full = df["Churn"]

    # Refit a fresh deployment pipeline on all available rows so the exported
    # preprocessing parameters and SGD coefficients describe the same model.
    deployment_pipeline = clone(sgd_model)
    deployment_pipeline.fit(X_full, y_full)

    fitted_preprocessor = deployment_pipeline.named_steps["preprocessor"]
    num_pipe = fitted_preprocessor.named_transformers_["num"]
    cat_pipe = fitted_preprocessor.named_transformers_["cat"]
    clf = deployment_pipeline.named_steps["clf"]

    imputer = num_pipe.named_steps["imputer"]
    scaler = num_pipe.named_steps["scaler"]
    ohe = cat_pipe.named_steps["onehot"]

    def json_scalar(value):
        if isinstance(value, np.generic):
            return value.item()
        return value

    categories = [
        [json_scalar(value) for value in category_values]
        for category_values in ohe.categories_
    ]

    artifact = {
        "numeric_features": list(numeric_features),
        "numeric_imputer_medians": imputer.statistics_.astype(float).tolist(),
        "numeric_means": scaler.mean_.astype(float).tolist(),
        "numeric_scales": scaler.scale_.astype(float).tolist(),
        "categorical_features": list(categorical_features),
        "categorical_imputer_modes": [json_scalar(value) for value in cat_pipe.named_steps["imputer"].statistics_],
        "categories": categories,
        "coef": clf.coef_.astype(float).tolist(),
        "intercept": clf.intercept_.astype(float).tolist(),
        "threshold": 0.5,
        "positive_class": "Churn",
    }

    return artifact


def save_gradient_plot(gradient_history, path="keras_gradient_magnitudes.svg"):
    """Save the gradient diagnostics as an SVG plot."""
    if not gradient_history:
        return None

    epochs = [row["epoch"] + 1 for row in gradient_history]
    series = {
        "Mean |gradient|": [row["grad_abs_mean"] for row in gradient_history],
        "Max |gradient|": [row["grad_abs_max"] for row in gradient_history],
        "Gradient std": [row["grad_std"] for row in gradient_history],
    }
    colors = {
        "Mean |gradient|": "#2563eb",
        "Max |gradient|": "#dc2626",
        "Gradient std": "#059669",
    }

    width, height = 900, 520
    left, right, top, bottom = 80, 30, 50, 80
    plot_width = width - left - right
    plot_height = height - top - bottom
    y_max = max(max(values) for values in series.values())
    y_max = y_max * 1.1 if y_max > 0 else 1.0
    x_min, x_max = min(epochs), max(epochs)
    x_span = max(x_max - x_min, 1)

    def xy(epoch, value):
        x = left + ((epoch - x_min) / x_span) * plot_width
        y = top + plot_height - (value / y_max) * plot_height
        return x, y

    lines = []
    for label, values in series.items():
        points = " ".join(
            f"{xy(epoch, value)[0]:.1f},{xy(epoch, value)[1]:.1f}"
            for epoch, value in zip(epochs, values)
        )
        lines.append(
            f'<polyline points="{points}" fill="none" '
            f'stroke="{colors[label]}" stroke-width="3" />'
        )

    x_ticks = []
    for epoch in epochs:
        if epoch == 1 or epoch == epochs[-1] or epoch % 5 == 0:
            x, _ = xy(epoch, 0)
            x_ticks.append(
                f'<line x1="{x:.1f}" y1="{top + plot_height}" '
                f'x2="{x:.1f}" y2="{top + plot_height + 6}" stroke="#444" />'
                f'<text x="{x:.1f}" y="{top + plot_height + 24}" '
                f'text-anchor="middle" font-size="12">{epoch}</text>'
            )

    y_ticks = []
    for frac in np.linspace(0, 1, 6):
        value = frac * y_max
        y = top + plot_height - frac * plot_height
        y_ticks.append(
            f'<line x1="{left - 6}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" '
            f'stroke="#444" />'
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" '
            f'y2="{y:.1f}" stroke="#e5e7eb" />'
            f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-size="12">{value:.4f}</text>'
        )

    legend = []
    legend_x = width - 240
    for index, label in enumerate(series):
        y = top + 20 + index * 22
        legend.append(
            f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 25}" y2="{y}" '
            f'stroke="{colors[label]}" stroke-width="3" />'
            f'<text x="{legend_x + 32}" y="{y + 4}" font-size="13">{label}</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white" />
  <text x="{width / 2}" y="28" text-anchor="middle" font-size="20" font-family="Arial">Keras Gradient Magnitudes Over Training</text>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#111" />
  <line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#111" />
  <text x="{left + plot_width / 2}" y="{height - 20}" text-anchor="middle" font-size="14" font-family="Arial">Epoch</text>
  <text x="20" y="{top + plot_height / 2}" text-anchor="middle" font-size="14" font-family="Arial" transform="rotate(-90 20 {top + plot_height / 2})">Gradient magnitude</text>
  {''.join(y_ticks)}
  {''.join(x_ticks)}
  {''.join(lines)}
  {''.join(legend)}
</svg>
"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(svg)

    return path


def interpret_gradient_behavior(gradient_history):
    """Summarize gradient magnitudes using heuristic thresholds, not a convergence proof."""
    if not gradient_history:
        return "No gradient history was recorded."

    means = np.array([row["grad_abs_mean"] for row in gradient_history], dtype=float)
    maxes = np.array([row["grad_abs_max"] for row in gradient_history], dtype=float)
    losses = np.array([row["logs"].get("loss", np.nan) for row in gradient_history])

    final_mean = means[-1]
    peak_max = np.max(maxes)
    ratio = final_mean / max(means[0], 1e-12)
    loss_delta = losses[-1] - losses[0] if np.all(np.isfinite(losses)) else np.nan

    if peak_max > 100 or ratio > 100:
        behavior = "exploding"
    elif final_mean < 1e-8 or ratio < 1e-3:
        behavior = "vanishing"
    else:
        behavior = "stable"

    loss_phrase = ""
    if np.isfinite(loss_delta):
        direction = "decreased" if loss_delta < 0 else "increased"
        loss_phrase = f" Training loss {direction} by {abs(loss_delta):.4f}."

    return (
        f"Gradients appear {behavior}: mean |gradient| changed from "
        f"{means[0]:.6f} to {final_mean:.6f}, with peak max |gradient| "
        f"{peak_max:.6f}.{loss_phrase}"
    )


#######################
# Optional AWS upload
def upload_json_to_s3(obj, bucket, key, region="us-east-1"):
    """Upload an optional deployment artifact using locally configured AWS credentials."""
    import boto3

    s3 = boto3.client("s3", region_name=region)
    body = json.dumps(obj).encode("utf-8")
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType="application/json",
    )
    print(f"Uploaded to s3://{bucket}/{key}")


def main():
    parser = argparse.ArgumentParser(description="Compare Telco churn models on a stratified validation split.")
    parser.add_argument("--input", type=Path, default=LOCAL_CSV)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "telco")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--aws-export", action="store_true", help="Optionally upload an SGD artifact to your S3 bucket.")
    parser.add_argument("--bucket", default=S3_BUCKET)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.aws_export and not args.bucket:
        parser.error("--aws-export requires --bucket or MODEL_BUCKET")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)
    tf.random.set_seed(42)

    # Step 0: Load data
    if not args.input.is_file():
        raise FileNotFoundError(f"CSV not found at {args.input}")

    print("Loading telco data...")
    df = load_telco(args.input)

    # Step 1: Preprocessor
    print("Building preprocessor...")
    preprocessor, X, y, num_feats, cat_feats = build_preprocessor(df)

    # Step 2: Scikit-Learn models
    print("Training sklearn models...")
    sk_results, splits, sgd_model, sgd_auc = train_sklearn_models(preprocessor, X, y)
    (X_train, X_val, y_train, y_val) = splits

    # sk_results should be sorted by AUC (descending)
    best_sklearn_name, best_sklearn_model, best_sklearn_auc, best_sklearn_acc = sk_results[0]

    # Step 3: Keras model
    print("Preparing numpy for Keras...")
    X_train_np, X_val_np, feature_names = make_numpy_for_keras(
        preprocessor, X_train, X_val
    )

    print("Building Keras model...")
    keras_model = build_keras_model(input_dim=X_train_np.shape[1])

    # pick a small fixed batch for gradient tracking
    sample_x = X_train_np[:256]
    sample_y = y_train.values[:256]
    grad_tracker = GradientTracker(keras_model, sample_x, sample_y)

    print("Training Keras model...")
    history = keras_model.fit(
        X_train_np,
        y_train.values,
        validation_data=(X_val_np, y_val.values),
        epochs=args.epochs,
        batch_size=256,
        callbacks=[grad_tracker],
        verbose=0,
    )
    print(
        "Keras training complete: "
        f"final train loss={history.history['loss'][-1]:.4f}, "
        f"final val loss={history.history['val_loss'][-1]:.4f}"
    )

    # Save gradient history locally for inspection
    with open(args.output_dir / "keras_gradients.json", "w") as f:
        json.dump(grad_tracker.history, f, indent=2)
    print("Saved gradient stats to keras_gradients.json")

    gradient_plot_path = save_gradient_plot(grad_tracker.history, args.output_dir / "keras_gradient_magnitudes.svg")
    if gradient_plot_path is not None:
        print(f"Saved gradient plot to {gradient_plot_path}")

    # Step 4: Evaluate Keras on validation set
    y_val_pred = keras_model.predict(X_val_np, verbose=0).ravel()
    keras_auc = roc_auc_score(y_val.values, y_val_pred)
    keras_acc = accuracy_score(y_val.values, (y_val_pred >= 0.5).astype(int))

    # Example: simple gradient quality score (for analysis)
    last_grad = grad_tracker.history[-1]
    grad_quality = {
        "last_epoch_grad_abs_mean": last_grad["grad_abs_mean"],
        "last_epoch_grad_abs_max": last_grad["grad_abs_max"],
        "last_epoch_grad_std": last_grad["grad_std"],
    }
    grad_interpretation = interpret_gradient_behavior(grad_tracker.history)

    print("\n=== MODEL SUMMARY (local) ===")
    print(
        f"Best sklearn: {best_sklearn_name} "
        f"AUC={best_sklearn_auc:.4f} ACC={best_sklearn_acc:.4f}"
    )
    print(f"Keras model : AUC={keras_auc:.4f} ACC={keras_acc:.4f}")
    print(f"SGD (for potential Lambda): AUC={sgd_auc:.4f}")
    print(f"Gradient quality (keras): {grad_quality}")
    print(f"Gradient interpretation: {grad_interpretation}")

    # Step 5: Apply the original course model-selection policy

    # Policy: prefer Keras only when it is competitive with the best sklearn
    # model and the gradient tracker does not flag vanishing/exploding behavior.
    keras_is_competitive = keras_auc >= best_sklearn_auc - 0.01
    gradients_are_healthy = "stable" in grad_interpretation
    if keras_is_competitive and gradients_are_healthy:
        final_model_name = "keras_model"
        final_auc = keras_auc
        final_acc = keras_acc
    else:
        final_model_name = best_sklearn_name
        final_auc = best_sklearn_auc
        final_acc = best_sklearn_acc

    print(
        f"\nFINAL CHOICE (for your report): {final_model_name} "
        f"AUC={final_auc:.4f} ACC={final_acc:.4f}"
    )

    # Step 6: Build leaderboard (for analysis/report)
    model_leaderboard = []
    for name, model_obj, auc_val, acc_val in sk_results:
        model_leaderboard.append(
            {
                "name": name,
                "type": "sklearn",
                "auc": float(auc_val),
                "accuracy": float(acc_val),
            }
        )

    # Add Keras to leaderboard
    model_leaderboard.append(
        {
            "name": "keras_mlp",
            "type": "keras",
            "auc": float(keras_auc),
            "accuracy": float(keras_acc),
            "grad_abs_mean": float(grad_tracker.history[-1]["grad_abs_mean"]),
            "grad_abs_max": float(grad_tracker.history[-1]["grad_abs_max"]),
            "grad_std": float(grad_tracker.history[-1]["grad_std"]),
            "gradient_interpretation": grad_interpretation,
        }
    )

    # At this point you can:
    #   - Save model_leaderboard to disk as JSON or CSV for inspection.
    #   - Create plots (e.g., bar charts of AUC/accuracy) in a separate script/notebook.
    with open(args.output_dir / "model_leaderboard_telco.json", "w") as f:
        json.dump(model_leaderboard, f, indent=2)
    print("Saved model_leaderboard_telco.json")

    comparison_df = pd.DataFrame(model_leaderboard)
    comparison_df = comparison_df.sort_values("auc", ascending=False)
    comparison_df.to_csv(args.output_dir / "model_comparison_telco.csv", index=False)
    print("Saved model_comparison_telco.csv")

    print("\n=== VALIDATION COMPARISON TABLE ===")
    print(
        comparison_df[["name", "type", "auc", "accuracy"]]
        .to_string(index=False, formatters={"auc": "{:.4f}".format, "accuracy": "{:.4f}".format})
    )

    # Step 7 (Bonus): Build lightweight artifact for AWS Lambda
    if args.aws_export:
        print("\nBuilding an optional lightweight artifact and uploading to S3.")

        artifact = build_lightweight_artifact(
            df,
            preprocessor,
            num_feats,
            cat_feats,
            sgd_model,
        )

        wrapped_artifact = {
            "artifact": artifact,
            "training_meta": {
                "final_choice": final_model_name,
                "final_choice_auc": float(final_auc),
                "final_choice_acc": float(final_acc),
                "best_sklearn": best_sklearn_name,
                "best_sklearn_auc": float(best_sklearn_auc),
                "best_sklearn_acc": float(best_sklearn_acc),
                "keras_auc": float(keras_auc),
                "keras_acc": float(keras_acc),
                "gradient_interpretation": grad_interpretation,
                "model_leaderboard": model_leaderboard,
            },
        }

        upload_json_to_s3(wrapped_artifact, args.bucket, S3_KEY_LIGHT, AWS_REGION)
        print("Uploaded lightweight artifact with metadata to S3.")
    else:
        print("Local run complete. Optional S3 export was not requested.")



if __name__ == "__main__":
    main()
