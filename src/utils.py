"""Reusable utilities for notebook and script workflows."""

from __future__ import annotations

import csv
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import setuptools
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
MANIFEST_PATH = FIGURES_DIR / "figure_manifest.csv"
QUANTILES = tf.constant([0.05, 0.50, 0.95], dtype=tf.float32)
MANIFEST_FIELDNAMES = ["Figure ID", "filename", "description"]


def _sanitize_filename(value: str) -> str:
    """Create a filesystem-safe filename stem."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def _figure_description(fig_name: str) -> str:
    """Create a simple human-readable description from a figure stem."""
    return fig_name.replace("_", " ").strip()


def _update_manifest(entries: list[dict[str, str]]) -> None:
    """Upsert figure metadata into the manifest."""
    existing: dict[tuple[str, str], dict[str, str]] = {}

    if MANIFEST_PATH.exists():
        with MANIFEST_PATH.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                existing[(row["Figure ID"], row["filename"])] = row

    for entry in entries:
        existing[(entry["Figure ID"], entry["filename"])] = entry

    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDNAMES)
        writer.writeheader()
        for _, row in sorted(existing.items()):
            writer.writerow(row)


def reset_figure_manifest() -> Path:
    """Clear the figure manifest so a notebook run starts fresh."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDNAMES)
        writer.writeheader()
    return MANIFEST_PATH


def save_publication_fig(fig_id: str, fig_name: str) -> tuple[Path, Path]:
    """Save the current matplotlib figure in publication-ready formats."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    safe_stem = _sanitize_filename(f"{fig_id}_{fig_name}")
    png_path = FIGURES_DIR / f"{safe_stem}.png"
    pdf_path = FIGURES_DIR / f"{safe_stem}.pdf"

    plt.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")

    description = _figure_description(fig_name)
    _update_manifest(
        [
            {"Figure ID": fig_id, "filename": png_path.name, "description": description},
            {"Figure ID": fig_id, "filename": pdf_path.name, "description": description},
        ]
    )

    return png_path, pdf_path


def save_metrics(df: pd.DataFrame, filename: str, *, index: bool = False) -> Path:
    """Persist a metrics dataframe to the outputs/metrics directory."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = METRICS_DIR / filename
    df.to_csv(output_path, index=index)
    return output_path


def enable_mps_acceleration():
    import tensorflow as tf

    mps_devices = tf.config.list_physical_devices("MPS")
    if mps_devices:
        tf.config.set_visible_devices(mps_devices, "MPS")
        print("✅ Apple Metal (MPS) acceleration enabled on M1/M2/M3")
        return True

    # TensorFlow Metal commonly exposes Apple Silicon acceleration as a GPU device.
    gpu_devices = tf.config.list_physical_devices("GPU")
    if gpu_devices:
        tf.config.set_visible_devices(gpu_devices, "GPU")
        print("✅ Apple Metal (MPS) acceleration enabled on M1/M2/M3")
        return True

    print("⚠️ No MPS device found – falling back to CPU")
    return False


@tf.keras.utils.register_keras_serializable(package="EnergyAI")
def quantile_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Pinball loss for 5th, 50th, and 95th percentile forecasts."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    if y_true.shape.rank == 1:
        y_true = tf.expand_dims(y_true, axis=-1)

    if y_true.shape.rank is not None and y_pred.shape.rank is not None:
        if y_true.shape.rank == y_pred.shape.rank - 1:
            y_true = tf.expand_dims(y_true, axis=-1)
    else:
        y_true = tf.cond(
            tf.equal(tf.rank(y_true), tf.rank(y_pred) - 1),
            lambda: tf.expand_dims(y_true, axis=-1),
            lambda: y_true,
        )

    error = y_true - y_pred
    loss = tf.maximum(QUANTILES * error, (QUANTILES - 1.0) * error)
    loss = tf.reduce_mean(loss, axis=-1)

    if loss.shape.rank is not None and loss.shape.rank > 1:
        reduce_axes = list(range(1, loss.shape.rank))
        loss = tf.reduce_mean(loss, axis=reduce_axes)
    elif loss.shape.rank is None:
        reduce_axes = tf.range(1, tf.rank(loss))
        loss = tf.cond(
            tf.size(reduce_axes) > 0,
            lambda: tf.reduce_mean(loss, axis=reduce_axes),
            lambda: loss,
        )

    return loss


def build_weighted_quantile_loss(
    *,
    quantile_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    name: str = "weighted_quantile_loss",
):
    """Create a weighted pinball loss over the fixed project quantiles."""
    weights = tf.constant(quantile_weights, dtype=tf.float32)

    @tf.keras.utils.register_keras_serializable(package="EnergyAI", name=name)
    def weighted_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true_cast = tf.cast(y_true, tf.float32)
        y_pred_cast = tf.cast(y_pred, tf.float32)

        if y_true_cast.shape.rank == 1:
            y_true_cast = tf.expand_dims(y_true_cast, axis=-1)

        if y_true_cast.shape.rank is not None and y_pred_cast.shape.rank is not None:
            if y_true_cast.shape.rank == y_pred_cast.shape.rank - 1:
                y_true_cast = tf.expand_dims(y_true_cast, axis=-1)
        else:
            y_true_cast = tf.cond(
                tf.equal(tf.rank(y_true_cast), tf.rank(y_pred_cast) - 1),
                lambda: tf.expand_dims(y_true_cast, axis=-1),
                lambda: y_true_cast,
            )

        error = y_true_cast - y_pred_cast
        loss = tf.maximum(QUANTILES * error, (QUANTILES - 1.0) * error)
        loss = loss * weights
        loss = tf.reduce_mean(loss, axis=-1)

        if loss.shape.rank is not None and loss.shape.rank > 1:
            reduce_axes = list(range(1, loss.shape.rank))
            loss = tf.reduce_mean(loss, axis=reduce_axes)
        elif loss.shape.rank is None:
            reduce_axes = tf.range(1, tf.rank(loss))
            loss = tf.cond(
                tf.size(reduce_axes) > 0,
                lambda: tf.reduce_mean(loss, axis=reduce_axes),
                lambda: loss,
            )

        return loss

    return weighted_loss


def t_confidence_interval_half_width(values, confidence: float = 0.95) -> float:
    """Return the t-based CI half width for a 1D sample."""
    sample = np.asarray(values, dtype=float)
    sample = sample[np.isfinite(sample)]
    n = sample.size
    if n <= 1:
        return 0.0

    from scipy.stats import t

    std = float(np.std(sample, ddof=1))
    se = std / np.sqrt(n)
    t_crit = float(t.ppf((1.0 + confidence) / 2.0, df=n - 1))
    return t_crit * se


def _inverse_transform_quantiles(
    scaler_target, y_pred_scaled: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inverse-transform quantile predictions one column at a time."""
    quantile_columns = [
        scaler_target.inverse_transform(y_pred_scaled[:, [idx]]).ravel()
        for idx in range(y_pred_scaled.shape[1])
    ]
    return tuple(quantile_columns)


def evaluate_model(
    model_name: str,
    history,
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler_target,
    start_time: float,
) -> dict[str, float | str]:
    """Evaluate a probabilistic model and return point plus interval metrics."""
    end_time = time.time()
    training_time = end_time - start_time
    print(f"\n--- Evaluation for {model_name} (Training Time: {training_time:.2f} seconds) ---")

    figure_map = {
        "Tuned Baseline LSTM": {
            "loss": ("Fig04", "lstm_training_history"),
            "pred": ("Fig05", "lstm_test_predictions"),
        },
        "Tuned CNN-LSTM": {
            "loss": ("Fig06", "cnn_lstm_training_history"),
            "pred": ("Fig07", "cnn_lstm_test_predictions"),
        },
        "Tuned Transformer": {
            "loss": ("Fig08", "transformer_training_history"),
            "pred": ("Fig09", "transformer_test_predictions"),
        },
    }
    model_slug = model_name.lower().replace(" ", "_").replace("-", "_")
    loss_fig = figure_map.get(model_name, {}).get(
        "loss", ("FigX", f"{model_slug}_training_history")
    )
    pred_fig = figure_map.get(model_name, {}).get(
        "pred", ("FigX", f"{model_slug}_test_predictions")
    )

    plt.figure(figsize=(12, 6))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title(f"{model_name} - Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Scaled Quantile Loss")
    plt.legend()
    plt.grid(True)
    save_publication_fig(*loss_fig)
    plt.show()

    y_pred_scaled = model.predict(X_test, verbose=0)
    y_actual = scaler_target.inverse_transform(np.asarray(y_test).reshape(-1, 1)).ravel()
    q05_pred, q50_pred, q95_pred = _inverse_transform_quantiles(scaler_target, y_pred_scaled)

    lower_bound = np.minimum(q05_pred, q95_pred)
    upper_bound = np.maximum(q05_pred, q95_pred)

    rmse = np.sqrt(mean_squared_error(y_actual, q50_pred))
    mae = mean_absolute_error(y_actual, q50_pred)
    r2 = r2_score(y_actual, q50_pred)
    cv_rmse = (rmse / y_actual.mean()) * 100
    picp = np.mean((y_actual >= lower_bound) & (y_actual <= upper_bound)) * 100
    mpiw = np.mean(upper_bound - lower_bound)

    print("\nPerformance Metrics on Test Set (original kWh scale):")
    print(f"Test RMSE: {rmse:.2f} kWh")
    print(f"Test MAE: {mae:.2f} kWh")
    print(f"Test R-squared: {r2:.4f}")
    print(f"Test CV(RMSE): {cv_rmse:.2f}%")
    print(f"Test PICP: {picp:.2f}%")
    print(f"Test MPIW: {mpiw:.2f} kWh")

    x_axis = np.arange(len(y_actual))
    plt.figure(figsize=(15, 6))
    plt.plot(x_axis, y_actual, label="Actual kWh", color="#0072B2", alpha=0.85)
    plt.plot(x_axis, q50_pred, label="Median Prediction (P50)", color="#D55E00", linestyle="--")
    plt.fill_between(
        x_axis,
        lower_bound,
        upper_bound,
        color="#56B4E9",
        alpha=0.25,
        label="5th-95th Percentile Interval",
    )
    plt.title(f"{model_name} - Probabilistic Test Forecasts")
    plt.xlabel("Time Step (Hour)")
    plt.ylabel("Total kWh")
    plt.legend()
    plt.grid(True)
    save_publication_fig(*pred_fig)
    plt.show()

    return {
        "Model": model_name,
        "Test RMSE (kWh)": rmse,
        "Test MAE (kWh)": mae,
        "Test R-squared": r2,
        "Test CV(RMSE) (%)": cv_rmse,
        "Test PICP (%)": picp,
        "Test MPIW (kWh)": mpiw,
        "Training Time (s)": training_time,
    }
