#!/usr/bin/env python3
"""Targeted strict/practical few-shot transfer-learning recipe search."""

from __future__ import annotations

import os
import ssl
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from zlib import adler32

import certifi
import setuptools
import keras_tuner as kt
import numpy as np
import pandas as pd
import tensorflow as tf
from meteostat import Base, Hourly, Point
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Conv1D, Dense, Dropout, LSTM, Reshape
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.chdir(PROJECT_ROOT)
os.environ["SSL_CERT_FILE"] = certifi.where()
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
Base.cache_dir = str((PROJECT_ROOT / ".meteostat").resolve())
Path(Base.cache_dir).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))

from src.utils import (  # noqa: E402
    build_weighted_quantile_loss,
    enable_mps_acceleration,
    quantile_loss,
    save_metrics,
    t_confidence_interval_half_width,
)


WEATHER_LOCATION = Point(13.7563, 100.5018)
HYPERPARAMETER_TUNING_DIR = (PROJECT_ROOT / "notebooks" / "hyperparameter_tuning").resolve()
TIME_STEPS = 24
FINAL_TRAINING_DAYS = [1, 3, 7, 14]
SELECTION_TRAINING_DAYS = [3, 14]
FINAL_WINDOWS_PER_BUDGET = 10
SELECTION_WINDOWS_PER_BUDGET = 3
BASE_SEED = 2026
MAX_EPOCHS = 60
STAGE2_EPOCHS = 15

CYCLICAL_COLUMNS = [
    "Hour_sin",
    "Hour_cos",
    "DayOfWeek_sin",
    "DayOfWeek_cos",
    "Month_sin",
    "Month_cos",
    "WeekOfYear_sin",
    "WeekOfYear_cos",
]
BINARY_COLUMNS = [
    "IsWeekend",
    "IsHoliday",
    "Indoor_Temp_mask",
    "Indoor_RH_mask",
    "Indoor_Lux_mask",
]
CONTINUOUS_COLUMNS = [
    "Outdoor_Temp",
    "Outdoor_RH",
    "Outdoor_WindSpeed",
    "Indoor_Temp_Avg",
    "Indoor_RH_Avg",
    "Indoor_Lux_Avg",
    "Total_kWh",
]
FEATURE_COLUMNS = CYCLICAL_COLUMNS + BINARY_COLUMNS + CONTINUOUS_COLUMNS
TARGET_COLUMN = "Total_kWh"

WEIGHTED_MEDIAN_LOSS = build_weighted_quantile_loss(
    quantile_weights=(1.0, 4.0, 1.0),
    name="weighted_quantile_loss_median4x",
)


HOLIDAYS = pd.to_datetime(
    [
        "2018-07-27",
        "2018-07-30",
        "2018-08-13",
        "2018-10-23",
        "2018-12-05",
        "2018-12-10",
        "2018-12-31",
        "2019-01-01",
        "2019-02-19",
        "2019-04-08",
        "2019-04-13",
        "2019-04-14",
        "2019-04-15",
        "2019-05-06",
        "2019-05-18",
        "2019-07-16",
        "2019-07-28",
        "2019-08-12",
        "2019-10-14",
        "2019-10-23",
        "2019-12-05",
        "2019-12-10",
        "2019-12-31",
    ]
)


@dataclass(frozen=True)
class PreprocessSpec:
    name: str
    scaler_kind: str
    family_aware: bool
    scale_on_full_pool: bool = False


@dataclass(frozen=True)
class OptimizationSpec:
    name: str
    learning_rate: float
    batch_cap: int
    patience: int
    clipnorm: float | None


@dataclass(frozen=True)
class LossSpec:
    name: str
    stage1_loss_name: str
    stage2_loss_name: str | None = None


STRICT_PREPROCESS_CANDIDATES = [
    PreprocessSpec("family_minmax", "minmax", family_aware=True),
    PreprocessSpec("family_standard", "standard", family_aware=True),
    PreprocessSpec("family_robust", "robust", family_aware=True),
]

# Reference baseline retained for reporting only; it is not eligible to win the
# primary strict protocol because the strict protocol must be family-aware.
REFERENCE_SCALE_ALL_PREPROCESS = PreprocessSpec("scale_all_minmax", "minmax", family_aware=False)

FINE_TUNE_STRATEGIES = [
    "head_only",
    "last_recurrent_block",
    "last_two_blocks",
    "full_model_tiny_lr",
]

LOSS_SPECS = [
    LossSpec("quantile_loss", "quantile"),
    LossSpec("weighted_median_quantile", "weighted"),
    LossSpec("two_stage_weighted_then_quantile", "weighted", "quantile"),
]


def stable_seed(*parts) -> int:
    return adler32("|".join(map(str, parts)).encode("utf-8")) % (2**31 - 1)


def set_all_seeds(seed: int) -> None:
    tf.keras.utils.set_random_seed(seed)
    np.random.seed(seed)


def create_sequences(features: np.ndarray, target: np.ndarray, time_steps: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for idx in range(len(features) - time_steps):
        X.append(features[idx : idx + time_steps])
        y.append(target[idx + time_steps])
    return np.asarray(X), np.asarray(y)


def aggregate_hourly_sensor_mean(df_floor: pd.DataFrame, sensor_columns: list[str], feature_name: str) -> pd.Series:
    if not sensor_columns:
        return pd.Series(np.nan, index=df_floor.resample("h").mean().index, name=feature_name)

    hourly_sensor_frame = df_floor[sensor_columns].resample("h").mean()
    sensor_values = hourly_sensor_frame.to_numpy(dtype=float)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        aggregated = np.nanmean(sensor_values, axis=1)

    aggregated[np.all(np.isnan(sensor_values), axis=1)] = np.nan
    return pd.Series(aggregated, index=hourly_sensor_frame.index, name=feature_name)


def fetch_hourly_weather(hourly_index: pd.DatetimeIndex) -> pd.DataFrame:
    weather_df = Hourly(
        WEATHER_LOCATION,
        hourly_index.min().to_pydatetime(),
        hourly_index.max().to_pydatetime(),
    ).fetch()

    weather_columns = ["temp", "rhum", "wspd"]
    weather_df = weather_df[weather_columns].copy()
    if getattr(weather_df.index, "tz", None) is not None:
        weather_df.index = weather_df.index.tz_convert("Asia/Bangkok").tz_localize(None)

    weather_df = weather_df.reindex(hourly_index)
    weather_df = weather_df.interpolate(method="time").ffill().bfill()
    weather_df.rename(
        columns={
            "temp": "Outdoor_Temp",
            "rhum": "Outdoor_RH",
            "wspd": "Outdoor_WindSpeed",
        },
        inplace=True,
    )
    return weather_df


def build_hourly_floor_dataframe(raw_floor_df: pd.DataFrame) -> pd.DataFrame:
    df_floor = raw_floor_df.copy()
    df_floor["Date"] = pd.to_datetime(df_floor["Date"])
    df_floor.set_index("Date", inplace=True)

    energy_columns = [col for col in df_floor.columns if col.endswith("(kW)")]
    df_energy = df_floor[energy_columns].copy()
    df_energy.ffill(inplace=True)
    df_energy.bfill(inplace=True)
    df_energy["Total_kW"] = df_energy.sum(axis=1)
    df_hourly = df_energy[["Total_kW"]].resample("h").mean()
    df_hourly.rename(columns={"Total_kW": "Total_kWh"}, inplace=True)

    weather_df = fetch_hourly_weather(df_hourly.index)

    indoor_temp_columns = [col for col in df_floor.columns if col.endswith("(degC)")]
    indoor_rh_columns = [col for col in df_floor.columns if col.endswith("(RH%)")]
    indoor_lux_columns = [col for col in df_floor.columns if col.endswith("(lux)")]

    indoor_features_df = pd.DataFrame(index=df_hourly.index)
    indoor_features_df["Indoor_Temp_Avg"] = aggregate_hourly_sensor_mean(
        df_floor, indoor_temp_columns, "Indoor_Temp_Avg"
    )
    indoor_features_df["Indoor_RH_Avg"] = aggregate_hourly_sensor_mean(
        df_floor, indoor_rh_columns, "Indoor_RH_Avg"
    )
    indoor_features_df["Indoor_Lux_Avg"] = aggregate_hourly_sensor_mean(
        df_floor, indoor_lux_columns, "Indoor_Lux_Avg"
    )
    indoor_features_df["Indoor_Temp_mask"] = indoor_features_df["Indoor_Temp_Avg"].notna().astype(int)
    indoor_features_df["Indoor_RH_mask"] = indoor_features_df["Indoor_RH_Avg"].notna().astype(int)
    indoor_features_df["Indoor_Lux_mask"] = indoor_features_df["Indoor_Lux_Avg"].notna().astype(int)
    indoor_features_df["Indoor_Temp_Avg"] = indoor_features_df["Indoor_Temp_Avg"].fillna(0.0)
    indoor_features_df["Indoor_RH_Avg"] = indoor_features_df["Indoor_RH_Avg"].fillna(0.0)
    indoor_features_df["Indoor_Lux_Avg"] = indoor_features_df["Indoor_Lux_Avg"].fillna(0.0)

    df_hourly = df_hourly.join(weather_df, how="left")
    df_hourly = df_hourly.join(indoor_features_df, how="left")
    return df_hourly


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    engineered = df.copy()
    engineered["Hour"] = engineered.index.hour
    engineered["DayOfWeek"] = engineered.index.dayofweek
    engineered["Month"] = engineered.index.month
    engineered["WeekOfYear"] = engineered.index.isocalendar().week.astype(int)
    engineered["IsWeekend"] = (engineered.index.dayofweek >= 5).astype(int)
    return engineered


def encode_cyclical(df: pd.DataFrame, col: str, max_val: int) -> pd.DataFrame:
    df[col + "_sin"] = np.sin(2 * np.pi * df[col] / max_val)
    df[col + "_cos"] = np.cos(2 * np.pi * df[col] / max_val)
    return df


def apply_feature_engineering(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, holidays: pd.DatetimeIndex
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = create_features(train_df)
    val_df = create_features(val_df)
    test_df = create_features(test_df)

    for frame in (train_df, val_df, test_df):
        frame["IsHoliday"] = frame.index.normalize().isin(holidays).astype(int)
        encode_cyclical(frame, "Hour", 24)
        encode_cyclical(frame, "DayOfWeek", 7)
        encode_cyclical(frame, "Month", 12)
        encode_cyclical(frame, "WeekOfYear", 52)

    return train_df, val_df, test_df


def sample_non_overlapping_window_starts(
    num_sequences: int, window_size: int, n_windows: int, seed: int
) -> list[int]:
    effective_span = window_size + TIME_STEPS - 1
    if effective_span > num_sequences + TIME_STEPS - 1:
        raise ValueError("Requested window is larger than the available number of sequences.")

    rng = np.random.default_rng(seed)
    candidate_starts = np.arange(0, num_sequences - window_size + 1)
    rng.shuffle(candidate_starts)

    selected_starts: list[int] = []
    for start in candidate_starts:
        start = int(start)
        if all(start + effective_span <= chosen or chosen + effective_span <= start for chosen in selected_starts):
            selected_starts.append(start)
        if len(selected_starts) == n_windows:
            break

    if len(selected_starts) < n_windows:
        raise ValueError("Unable to find enough non-overlapping few-shot windows for the requested budget.")

    return sorted(selected_starts)


def scaler_from_kind(kind: str):
    if kind == "minmax":
        return MinMaxScaler()
    if kind == "standard":
        return StandardScaler()
    if kind == "robust":
        return RobustScaler()
    raise ValueError(f"Unsupported scaler kind: {kind}")


def fit_feature_scaler(
    preprocess_spec: PreprocessSpec, train_window_df: pd.DataFrame, train_pool_df: pd.DataFrame | None = None
):
    scaler = scaler_from_kind(preprocess_spec.scaler_kind)
    fit_frame = train_pool_df if preprocess_spec.scale_on_full_pool and train_pool_df is not None else train_window_df
    if preprocess_spec.family_aware:
        scaler.fit(fit_frame[CONTINUOUS_COLUMNS])
    else:
        scaler.fit(fit_frame[FEATURE_COLUMNS])
    return scaler


def transform_feature_frame(df: pd.DataFrame, preprocess_spec: PreprocessSpec, scaler) -> np.ndarray:
    feature_frame = df[FEATURE_COLUMNS].copy()
    if preprocess_spec.family_aware:
        feature_frame.loc[:, CONTINUOUS_COLUMNS] = scaler.transform(feature_frame[CONTINUOUS_COLUMNS])
        return feature_frame.to_numpy(dtype=float)

    return np.asarray(scaler.transform(feature_frame), dtype=float)


def prepare_window_splits(
    preprocess_spec: PreprocessSpec,
    train_window_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_pool_df: pd.DataFrame | None = None,
):
    feature_scaler = fit_feature_scaler(preprocess_spec, train_window_df, train_pool_df=train_pool_df)
    target_scaler = MinMaxScaler()
    target_scaler.fit(train_window_df[[TARGET_COLUMN]])

    X_train = transform_feature_frame(train_window_df, preprocess_spec, feature_scaler)
    y_train = target_scaler.transform(train_window_df[[TARGET_COLUMN]]).ravel()

    X_val = transform_feature_frame(val_df, preprocess_spec, feature_scaler)
    y_val = target_scaler.transform(val_df[[TARGET_COLUMN]]).ravel()

    X_test = transform_feature_frame(test_df, preprocess_spec, feature_scaler)
    y_test = target_scaler.transform(test_df[[TARGET_COLUMN]]).ravel()

    X_train_seq, y_train_seq = create_sequences(X_train, y_train, TIME_STEPS)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, TIME_STEPS)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, TIME_STEPS)
    return X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_test_seq, y_test_seq, target_scaler


class CNNLSTMHyperModel(kt.HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):
        model = Sequential(name="CNN_LSTM")
        hp_filters = hp.Int("filters", min_value=64, max_value=128, step=32)
        hp_lstm_units_1 = hp.Int("lstm_units_1", min_value=100, max_value=200, step=50)
        hp_lstm_units_2 = hp.Int("lstm_units_2", min_value=50, max_value=150, step=50)
        hp_dropout = hp.Float("dropout", min_value=0.2, max_value=0.4, step=0.1)
        hp_learning_rate = hp.Choice("learning_rate", values=[1e-3, 5e-4])

        model.add(
            Conv1D(
                filters=hp_filters,
                kernel_size=3,
                activation="relu",
                input_shape=self.input_shape,
                padding="same",
            )
        )
        model.add(LSTM(units=hp_lstm_units_1, return_sequences=True))
        model.add(Dropout(hp_dropout))
        model.add(LSTM(units=hp_lstm_units_2, return_sequences=False))
        model.add(Dropout(hp_dropout))
        model.add(Dense(3, name="quantile_output"))
        model.compile(optimizer=Adam(learning_rate=hp_learning_rate), loss=quantile_loss)
        return model


def load_source_champion(input_shape):
    tuner = kt.BayesianOptimization(
        CNNLSTMHyperModel(input_shape),
        objective="val_loss",
        max_trials=10,
        executions_per_trial=1,
        directory=str(HYPERPARAMETER_TUNING_DIR),
        project_name="cnn_lstm_tuning_quantile",
        overwrite=False,
        seed=42,
    )
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.get_best_models(num_models=1)[0]
    champion_params = {
        "filters": best_hps.get("filters"),
        "lstm_units_1": best_hps.get("lstm_units_1"),
        "lstm_units_2": best_hps.get("lstm_units_2"),
        "dropout": best_hps.get("dropout"),
        "learning_rate": best_hps.get("learning_rate"),
    }
    return best_model, champion_params


def build_probabilistic_champion_cnn_lstm(champion_params: dict[str, float], input_shape):
    model = Sequential(name="Champion_CNN_LSTM")
    model.add(
        Conv1D(
            filters=champion_params["filters"],
            kernel_size=3,
            activation="relu",
            input_shape=input_shape,
            padding="same",
        )
    )
    model.add(LSTM(champion_params["lstm_units_1"], return_sequences=True))
    model.add(Dropout(champion_params["dropout"]))
    model.add(LSTM(champion_params["lstm_units_2"], return_sequences=False))
    model.add(Dropout(champion_params["dropout"]))
    model.add(Dense(3, name="quantile_output"))
    return model


def configure_transfer_trainable_layers(model, strategy: str) -> None:
    weighted_layers = [layer for layer in model.layers if layer.weights]
    for layer in model.layers:
        layer.trainable = False

    if strategy == "head_only":
        trainable_layers = weighted_layers[-1:]
    elif strategy == "last_recurrent_block":
        trainable_layers = weighted_layers[-2:]
    elif strategy == "last_two_blocks":
        trainable_layers = weighted_layers[-3:]
    elif strategy == "full_model_tiny_lr":
        trainable_layers = weighted_layers
    else:
        raise ValueError(f"Unsupported fine-tuning strategy: {strategy}")

    for layer in trainable_layers:
        layer.trainable = True


def make_optimizer(learning_rate: float, clipnorm: float | None) -> Adam:
    kwargs = {}
    if clipnorm is not None:
        kwargs["clipnorm"] = clipnorm
    return Adam(learning_rate=learning_rate, **kwargs)


def resolve_loss(loss_spec: LossSpec, stage: int):
    if stage == 1:
        if loss_spec.stage1_loss_name == "quantile":
            return quantile_loss
        if loss_spec.stage1_loss_name == "weighted":
            return WEIGHTED_MEDIAN_LOSS
    if stage == 2 and loss_spec.stage2_loss_name is not None:
        if loss_spec.stage2_loss_name == "quantile":
            return quantile_loss
        if loss_spec.stage2_loss_name == "weighted":
            return WEIGHTED_MEDIAN_LOSS
    raise ValueError(f"Unsupported loss stage resolution for {loss_spec}")


def run_training_phase(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    *,
    learning_rate: float,
    batch_cap: int,
    patience: int,
    clipnorm: float | None,
    epochs: int,
    loss_fn,
):
    batch_size = max(1, min(batch_cap, len(X_train)))
    model.compile(
        optimizer=make_optimizer(learning_rate=learning_rate, clipnorm=clipnorm),
        loss=loss_fn,
    )
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(2, patience // 2), min_lr=1e-5),
        ],
        shuffle=False,
        verbose=0,
    )
    return float(min(history.history["val_loss"]))


def inverse_transform_median_prediction(model, X, target_scaler) -> np.ndarray:
    y_pred_scaled = np.asarray(model.predict(X, verbose=0))
    y_pred_median_scaled = y_pred_scaled[:, 1].reshape(-1, 1)
    return target_scaler.inverse_transform(y_pred_median_scaled).ravel()


def compute_rmse(model, X_eval, y_eval, target_scaler) -> float:
    y_pred = inverse_transform_median_prediction(model, X_eval, target_scaler)
    y_true = target_scaler.inverse_transform(y_eval.reshape(-1, 1)).ravel()
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def train_scratch_model(
    champion_params,
    input_shape,
    X_train,
    y_train,
    X_val,
    y_val,
    optim_spec: OptimizationSpec,
    loss_spec: LossSpec,
    *,
    run_seed: int,
):
    set_all_seeds(run_seed)
    model = build_probabilistic_champion_cnn_lstm(champion_params, input_shape)
    run_training_phase(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        learning_rate=optim_spec.learning_rate,
        batch_cap=optim_spec.batch_cap,
        patience=optim_spec.patience,
        clipnorm=optim_spec.clipnorm,
        epochs=MAX_EPOCHS,
        loss_fn=resolve_loss(loss_spec, stage=1),
    )
    if loss_spec.stage2_loss_name is not None:
        run_training_phase(
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            learning_rate=max(1e-5, optim_spec.learning_rate * 0.5),
            batch_cap=optim_spec.batch_cap,
            patience=max(3, optim_spec.patience // 2),
            clipnorm=optim_spec.clipnorm,
            epochs=STAGE2_EPOCHS,
            loss_fn=resolve_loss(loss_spec, stage=2),
        )
    return model


def train_transfer_model(
    source_model,
    champion_params,
    input_shape,
    X_train,
    y_train,
    X_val,
    y_val,
    optim_spec: OptimizationSpec,
    loss_spec: LossSpec,
    fine_tune_strategy: str,
    *,
    run_seed: int,
):
    set_all_seeds(run_seed)
    model = build_probabilistic_champion_cnn_lstm(champion_params, input_shape)
    model.set_weights(source_model.get_weights())
    configure_transfer_trainable_layers(model, fine_tune_strategy)

    transfer_lr = optim_spec.learning_rate
    if fine_tune_strategy == "full_model_tiny_lr":
        transfer_lr = min(transfer_lr, 5e-5)

    run_training_phase(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        learning_rate=transfer_lr,
        batch_cap=optim_spec.batch_cap,
        patience=optim_spec.patience,
        clipnorm=optim_spec.clipnorm,
        epochs=MAX_EPOCHS,
        loss_fn=resolve_loss(loss_spec, stage=1),
    )
    if loss_spec.stage2_loss_name is not None:
        run_training_phase(
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            learning_rate=max(1e-5, transfer_lr * 0.5),
            batch_cap=optim_spec.batch_cap,
            patience=max(3, optim_spec.patience // 2),
            clipnorm=optim_spec.clipnorm,
            epochs=STAGE2_EPOCHS,
            loss_fn=resolve_loss(loss_spec, stage=2),
        )
    return model


def evaluate_recipe(
    *,
    protocol_name: str,
    preprocess_spec: PreprocessSpec,
    fine_tune_strategy: str,
    optim_spec: OptimizationSpec,
    loss_spec: LossSpec,
    budgets: list[int],
    windows_per_budget: int,
    eval_split: str,
    source_model,
    champion_params,
    train_pool_df_f4,
    val_df_f4,
    test_df_f4,
):
    per_window_records = []
    summary_records = []
    train_pool_sequence_count = len(train_pool_df_f4) - TIME_STEPS

    for days in budgets:
        budget_sequences = days * 24
        budget_rows = budget_sequences + TIME_STEPS
        window_starts = sample_non_overlapping_window_starts(
            num_sequences=train_pool_sequence_count,
            window_size=budget_sequences,
            n_windows=FINAL_WINDOWS_PER_BUDGET,
            seed=BASE_SEED + days,
        )[:windows_per_budget]

        scratch_eval_rmses: list[float] = []
        transfer_eval_rmses: list[float] = []
        improvement_rmses: list[float] = []

        for window_idx, start_idx in enumerate(window_starts, start=1):
            end_idx = start_idx + budget_rows
            train_window_df = train_pool_df_f4.iloc[start_idx:end_idx].copy()

            X_train, y_train, X_val, y_val, X_test, y_test, target_scaler = prepare_window_splits(
                preprocess_spec,
                train_window_df,
                val_df_f4,
                test_df_f4,
                train_pool_df=train_pool_df_f4,
            )
            input_shape = (X_train.shape[1], X_train.shape[2])
            scratch_seed = stable_seed(protocol_name, preprocess_spec.name, fine_tune_strategy, optim_spec.name, loss_spec.name, days, window_idx, "scratch")
            transfer_seed = stable_seed(protocol_name, preprocess_spec.name, fine_tune_strategy, optim_spec.name, loss_spec.name, days, window_idx, "transfer")

            scratch_model = train_scratch_model(
                champion_params,
                input_shape,
                X_train,
                y_train,
                X_val,
                y_val,
                optim_spec,
                loss_spec,
                run_seed=scratch_seed,
            )
            transfer_model = train_transfer_model(
                source_model,
                champion_params,
                input_shape,
                X_train,
                y_train,
                X_val,
                y_val,
                optim_spec,
                loss_spec,
                fine_tune_strategy,
                run_seed=transfer_seed,
            )

            scratch_val_rmse = compute_rmse(scratch_model, X_val, y_val, target_scaler)
            transfer_val_rmse = compute_rmse(transfer_model, X_val, y_val, target_scaler)
            scratch_test_rmse = compute_rmse(scratch_model, X_test, y_test, target_scaler)
            transfer_test_rmse = compute_rmse(transfer_model, X_test, y_test, target_scaler)

            chosen_scratch_rmse = scratch_val_rmse if eval_split == "val" else scratch_test_rmse
            chosen_transfer_rmse = transfer_val_rmse if eval_split == "val" else transfer_test_rmse

            scratch_eval_rmses.append(chosen_scratch_rmse)
            transfer_eval_rmses.append(chosen_transfer_rmse)
            improvement_rmses.append(chosen_scratch_rmse - chosen_transfer_rmse)

            per_window_records.append(
                {
                    "Protocol": protocol_name,
                    "Preprocess": preprocess_spec.name,
                    "Fine-Tune Strategy": fine_tune_strategy,
                    "Optimization": optim_spec.name,
                    "Loss Strategy": loss_spec.name,
                    "Evaluation Split": eval_split,
                    "Training Days": days,
                    "Window Index": window_idx,
                    "Window Start": start_idx,
                    "Scratch Validation RMSE (kWh)": scratch_val_rmse,
                    "Transfer Validation RMSE (kWh)": transfer_val_rmse,
                    "Scratch Test RMSE (kWh)": scratch_test_rmse,
                    "Transfer Test RMSE (kWh)": transfer_test_rmse,
                    "Selected Scratch RMSE (kWh)": chosen_scratch_rmse,
                    "Selected Transfer RMSE (kWh)": chosen_transfer_rmse,
                    "Selected Improvement (kWh)": chosen_scratch_rmse - chosen_transfer_rmse,
                }
            )

        summary_records.append(
            {
                "Protocol": protocol_name,
                "Preprocess": preprocess_spec.name,
                "Fine-Tune Strategy": fine_tune_strategy,
                "Optimization": optim_spec.name,
                "Loss Strategy": loss_spec.name,
                "Evaluation Split": eval_split,
                "Training Days": days,
                "Windows": len(window_starts),
                "Scratch RMSE Mean (kWh)": float(np.mean(scratch_eval_rmses)),
                "Scratch RMSE Std (kWh)": float(np.std(scratch_eval_rmses, ddof=1)) if len(scratch_eval_rmses) > 1 else 0.0,
                "Scratch RMSE 95% CI Half Width (kWh)": float(t_confidence_interval_half_width(scratch_eval_rmses)),
                "Transfer RMSE Mean (kWh)": float(np.mean(transfer_eval_rmses)),
                "Transfer RMSE Std (kWh)": float(np.std(transfer_eval_rmses, ddof=1)) if len(transfer_eval_rmses) > 1 else 0.0,
                "Transfer RMSE 95% CI Half Width (kWh)": float(t_confidence_interval_half_width(transfer_eval_rmses)),
                "Paired Improvement Mean (kWh)": float(np.mean(improvement_rmses)),
                "Paired Improvement Std (kWh)": float(np.std(improvement_rmses, ddof=1)) if len(improvement_rmses) > 1 else 0.0,
                "Paired Improvement 95% CI Half Width (kWh)": float(t_confidence_interval_half_width(improvement_rmses)),
            }
        )

    per_window_df = pd.DataFrame(per_window_records)
    summary_df = pd.DataFrame(summary_records)
    overall_summary = {
        "Protocol": protocol_name,
        "Preprocess": preprocess_spec.name,
        "Fine-Tune Strategy": fine_tune_strategy,
        "Optimization": optim_spec.name,
        "Loss Strategy": loss_spec.name,
        "Evaluation Split": eval_split,
        "Training Days": "ALL",
        "Windows": int(len(per_window_df)),
        "Scratch RMSE Mean (kWh)": float(per_window_df["Selected Scratch RMSE (kWh)"].mean()),
        "Scratch RMSE Std (kWh)": float(per_window_df["Selected Scratch RMSE (kWh)"].std(ddof=1)) if len(per_window_df) > 1 else 0.0,
        "Scratch RMSE 95% CI Half Width (kWh)": float(t_confidence_interval_half_width(per_window_df["Selected Scratch RMSE (kWh)"])),
        "Transfer RMSE Mean (kWh)": float(per_window_df["Selected Transfer RMSE (kWh)"].mean()),
        "Transfer RMSE Std (kWh)": float(per_window_df["Selected Transfer RMSE (kWh)"].std(ddof=1)) if len(per_window_df) > 1 else 0.0,
        "Transfer RMSE 95% CI Half Width (kWh)": float(t_confidence_interval_half_width(per_window_df["Selected Transfer RMSE (kWh)"])),
        "Paired Improvement Mean (kWh)": float(per_window_df["Selected Improvement (kWh)"].mean()),
        "Paired Improvement Std (kWh)": float(per_window_df["Selected Improvement (kWh)"].std(ddof=1)) if len(per_window_df) > 1 else 0.0,
        "Paired Improvement 95% CI Half Width (kWh)": float(t_confidence_interval_half_width(per_window_df["Selected Improvement (kWh)"])),
    }
    summary_df = pd.concat([summary_df, pd.DataFrame([overall_summary])], ignore_index=True)
    return pd.DataFrame(per_window_records), summary_df


def summarize_stage_candidate(stage_name: str, summary_df: pd.DataFrame) -> dict[str, object]:
    overall_row = summary_df.loc[summary_df["Training Days"] == "ALL"].iloc[0]
    return {
        "Stage": stage_name,
        "Protocol": overall_row["Protocol"],
        "Preprocess": overall_row["Preprocess"],
        "Fine-Tune Strategy": overall_row["Fine-Tune Strategy"],
        "Optimization": overall_row["Optimization"],
        "Loss Strategy": overall_row["Loss Strategy"],
        "Evaluation Split": overall_row["Evaluation Split"],
        "Validation Transfer RMSE Mean (kWh)": overall_row["Transfer RMSE Mean (kWh)"],
        "Validation Transfer RMSE Std (kWh)": overall_row["Transfer RMSE Std (kWh)"],
        "Validation Transfer RMSE 95% CI Half Width (kWh)": overall_row["Transfer RMSE 95% CI Half Width (kWh)"],
        "Validation Scratch RMSE Mean (kWh)": overall_row["Scratch RMSE Mean (kWh)"],
        "Validation Paired Improvement Mean (kWh)": overall_row["Paired Improvement Mean (kWh)"],
        "Windows": overall_row["Windows"],
    }


def main() -> None:
    enable_mps_acceleration()
    set_all_seeds(42)

    source_model, champion_params = load_source_champion((TIME_STEPS, len(FEATURE_COLUMNS)))

    floor4 = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "Floor4.csv")
    df_hourly_f4 = build_hourly_floor_dataframe(floor4)
    n_total_f4 = len(df_hourly_f4)
    train_end_idx_f4 = int(n_total_f4 * 0.70)
    val_end_idx_f4 = int(n_total_f4 * 0.85)

    train_pool_df_f4 = df_hourly_f4.iloc[:train_end_idx_f4].copy()
    val_df_f4 = df_hourly_f4.iloc[train_end_idx_f4:val_end_idx_f4].copy()
    test_df_f4 = df_hourly_f4.iloc[val_end_idx_f4:].copy()
    train_pool_df_f4, val_df_f4, test_df_f4 = apply_feature_engineering(
        train_pool_df_f4,
        val_df_f4,
        test_df_f4,
        HOLIDAYS,
    )

    base_lr = float(champion_params["learning_rate"])
    optimization_candidates = [
        OptimizationSpec("default", learning_rate=base_lr, batch_cap=64, patience=8, clipnorm=None),
        OptimizationSpec("small_batch_longer_patience", learning_rate=min(base_lr, 5e-4), batch_cap=32, patience=12, clipnorm=None),
        OptimizationSpec("low_lr_clipped", learning_rate=1e-4, batch_cap=32, patience=12, clipnorm=1.0),
    ]

    strict_candidate_rows = []
    stage_winners = []

    # Reference baseline: optimistic "scale all" preprocessing kept only as an
    # auxiliary comparison and never allowed to win the strict protocol.
    _, reference_baseline_summary_df = evaluate_recipe(
        protocol_name="reference_scale_all_baseline",
        preprocess_spec=REFERENCE_SCALE_ALL_PREPROCESS,
        fine_tune_strategy="head_only",
        optim_spec=optimization_candidates[0],
        loss_spec=LOSS_SPECS[0],
        budgets=SELECTION_TRAINING_DAYS,
        windows_per_budget=SELECTION_WINDOWS_PER_BUDGET,
        eval_split="val",
        source_model=source_model,
        champion_params=champion_params,
        train_pool_df_f4=train_pool_df_f4,
        val_df_f4=val_df_f4,
        test_df_f4=test_df_f4,
    )
    save_metrics(reference_baseline_summary_df, "stage1_scale_all_minmax_validation_summary.csv")

    # Stage 1: strict preprocessing / scaler comparison.
    stage1_candidates = []
    for preprocess_spec in STRICT_PREPROCESS_CANDIDATES:
        _, summary_df = evaluate_recipe(
            protocol_name="strict_primary",
            preprocess_spec=preprocess_spec,
            fine_tune_strategy="head_only",
            optim_spec=optimization_candidates[0],
            loss_spec=LOSS_SPECS[0],
            budgets=SELECTION_TRAINING_DAYS,
            windows_per_budget=SELECTION_WINDOWS_PER_BUDGET,
            eval_split="val",
            source_model=source_model,
            champion_params=champion_params,
            train_pool_df_f4=train_pool_df_f4,
            val_df_f4=val_df_f4,
            test_df_f4=test_df_f4,
        )
        candidate_summary = summarize_stage_candidate("stage1_preprocess", summary_df)
        strict_candidate_rows.append(candidate_summary)
        stage1_candidates.append((preprocess_spec, candidate_summary))
        save_metrics(summary_df, f"stage1_{preprocess_spec.name}_validation_summary.csv")

    best_preprocess_spec = min(
        stage1_candidates,
        key=lambda item: item[1]["Validation Transfer RMSE Mean (kWh)"],
    )[0]
    stage_winners.append({"Stage": "stage1_preprocess", "Winner": best_preprocess_spec.name})

    # Stage 2: fine-tuning strategy comparison.
    stage2_candidates = []
    for fine_tune_strategy in FINE_TUNE_STRATEGIES:
        _, summary_df = evaluate_recipe(
            protocol_name="strict_primary",
            preprocess_spec=best_preprocess_spec,
            fine_tune_strategy=fine_tune_strategy,
            optim_spec=optimization_candidates[0],
            loss_spec=LOSS_SPECS[0],
            budgets=SELECTION_TRAINING_DAYS,
            windows_per_budget=SELECTION_WINDOWS_PER_BUDGET,
            eval_split="val",
            source_model=source_model,
            champion_params=champion_params,
            train_pool_df_f4=train_pool_df_f4,
            val_df_f4=val_df_f4,
            test_df_f4=test_df_f4,
        )
        candidate_summary = summarize_stage_candidate("stage2_fine_tuning", summary_df)
        strict_candidate_rows.append(candidate_summary)
        stage2_candidates.append((fine_tune_strategy, candidate_summary))
        save_metrics(summary_df, f"stage2_{fine_tune_strategy}_validation_summary.csv")

    best_fine_tune_strategy = min(
        stage2_candidates,
        key=lambda item: item[1]["Validation Transfer RMSE Mean (kWh)"],
    )[0]
    stage_winners.append({"Stage": "stage2_fine_tuning", "Winner": best_fine_tune_strategy})

    # Stage 3: optimization comparison.
    stage3_candidates = []
    for optim_spec in optimization_candidates:
        _, summary_df = evaluate_recipe(
            protocol_name="strict_primary",
            preprocess_spec=best_preprocess_spec,
            fine_tune_strategy=best_fine_tune_strategy,
            optim_spec=optim_spec,
            loss_spec=LOSS_SPECS[0],
            budgets=SELECTION_TRAINING_DAYS,
            windows_per_budget=SELECTION_WINDOWS_PER_BUDGET,
            eval_split="val",
            source_model=source_model,
            champion_params=champion_params,
            train_pool_df_f4=train_pool_df_f4,
            val_df_f4=val_df_f4,
            test_df_f4=test_df_f4,
        )
        candidate_summary = summarize_stage_candidate("stage3_optimization", summary_df)
        strict_candidate_rows.append(candidate_summary)
        stage3_candidates.append((optim_spec, candidate_summary))
        save_metrics(summary_df, f"stage3_{optim_spec.name}_validation_summary.csv")

    best_optim_spec = min(
        stage3_candidates,
        key=lambda item: item[1]["Validation Transfer RMSE Mean (kWh)"],
    )[0]
    stage_winners.append({"Stage": "stage3_optimization", "Winner": best_optim_spec.name})

    # Stage 4: loss alignment comparison.
    stage4_candidates = []
    for loss_spec in LOSS_SPECS:
        _, summary_df = evaluate_recipe(
            protocol_name="strict_primary",
            preprocess_spec=best_preprocess_spec,
            fine_tune_strategy=best_fine_tune_strategy,
            optim_spec=best_optim_spec,
            loss_spec=loss_spec,
            budgets=SELECTION_TRAINING_DAYS,
            windows_per_budget=SELECTION_WINDOWS_PER_BUDGET,
            eval_split="val",
            source_model=source_model,
            champion_params=champion_params,
            train_pool_df_f4=train_pool_df_f4,
            val_df_f4=val_df_f4,
            test_df_f4=test_df_f4,
        )
        candidate_summary = summarize_stage_candidate("stage4_loss_alignment", summary_df)
        strict_candidate_rows.append(candidate_summary)
        stage4_candidates.append((loss_spec, candidate_summary))
        save_metrics(summary_df, f"stage4_{loss_spec.name}_validation_summary.csv")

    best_loss_spec = min(
        stage4_candidates,
        key=lambda item: item[1]["Validation Transfer RMSE Mean (kWh)"],
    )[0]
    stage_winners.append({"Stage": "stage4_loss_alignment", "Winner": best_loss_spec.name})

    strict_candidate_df = pd.DataFrame(strict_candidate_rows)
    save_metrics(strict_candidate_df, "strict_protocol_candidate_recipes.csv")
    save_metrics(pd.DataFrame(stage_winners), "strict_protocol_stage_winners.csv")

    # Auxiliary post-selection reference: compare the locked downstream recipe
    # against the old scale-all baseline on validation only.
    _, reference_locked_recipe_summary_df = evaluate_recipe(
        protocol_name="reference_scale_all_locked_recipe",
        preprocess_spec=REFERENCE_SCALE_ALL_PREPROCESS,
        fine_tune_strategy=best_fine_tune_strategy,
        optim_spec=best_optim_spec,
        loss_spec=best_loss_spec,
        budgets=SELECTION_TRAINING_DAYS,
        windows_per_budget=SELECTION_WINDOWS_PER_BUDGET,
        eval_split="val",
        source_model=source_model,
        champion_params=champion_params,
        train_pool_df_f4=train_pool_df_f4,
        val_df_f4=val_df_f4,
        test_df_f4=test_df_f4,
    )
    save_metrics(reference_locked_recipe_summary_df, "reference_scale_all_locked_recipe_validation_summary.csv")

    # Final strict test evaluation.
    strict_per_window_df, strict_summary_df = evaluate_recipe(
        protocol_name="strict_primary",
        preprocess_spec=best_preprocess_spec,
        fine_tune_strategy=best_fine_tune_strategy,
        optim_spec=best_optim_spec,
        loss_spec=best_loss_spec,
        budgets=FINAL_TRAINING_DAYS,
        windows_per_budget=FINAL_WINDOWS_PER_BUDGET,
        eval_split="test",
        source_model=source_model,
        champion_params=champion_params,
        train_pool_df_f4=train_pool_df_f4,
        val_df_f4=val_df_f4,
        test_df_f4=test_df_f4,
    )
    save_metrics(strict_per_window_df, "strict_best_recipe_per_window_test.csv")
    save_metrics(strict_summary_df, "strict_best_recipe_test_summary.csv")

    # Secondary practical protocol: full-pool continuous feature scaler only.
    practical_preprocess = PreprocessSpec(
        name=f"{best_preprocess_spec.name}_full_pool_continuous_scaler",
        scaler_kind=best_preprocess_spec.scaler_kind,
        family_aware=True,
        scale_on_full_pool=True,
    )
    practical_per_window_df, practical_summary_df = evaluate_recipe(
        protocol_name="practical_unlabeled_feature_scaler",
        preprocess_spec=practical_preprocess,
        fine_tune_strategy=best_fine_tune_strategy,
        optim_spec=best_optim_spec,
        loss_spec=best_loss_spec,
        budgets=FINAL_TRAINING_DAYS,
        windows_per_budget=FINAL_WINDOWS_PER_BUDGET,
        eval_split="test",
        source_model=source_model,
        champion_params=champion_params,
        train_pool_df_f4=train_pool_df_f4,
        val_df_f4=val_df_f4,
        test_df_f4=test_df_f4,
    )
    save_metrics(practical_per_window_df, "practical_protocol_per_window_test.csv")
    save_metrics(practical_summary_df, "practical_protocol_test_summary.csv")

    merged_comparison = strict_summary_df.merge(
        practical_summary_df,
        on="Training Days",
        suffixes=("_Strict", "_Practical"),
    )
    merged_comparison["Transfer RMSE Gain from Practical (kWh)"] = (
        merged_comparison["Transfer RMSE Mean (kWh)_Strict"]
        - merged_comparison["Transfer RMSE Mean (kWh)_Practical"]
    )
    save_metrics(merged_comparison, "practical_vs_strict_comparison.csv")

    print("\n=== Validation-Selected Best Strict Recipe ===")
    print(
        {
            "Preprocess": best_preprocess_spec.name,
            "Fine-Tune Strategy": best_fine_tune_strategy,
            "Optimization": best_optim_spec.name,
            "Loss Strategy": best_loss_spec.name,
        }
    )
    print("\n=== Strict Candidate Comparison Table ===")
    print(strict_candidate_df.to_string(index=False))
    print("\n=== Final Strict Test Summary ===")
    print(strict_summary_df.to_string(index=False))
    print("\n=== Practical Secondary Protocol Summary ===")
    print(practical_summary_df.to_string(index=False))
    print("\n=== Practical vs Strict Delta ===")
    print(merged_comparison.to_string(index=False))


if __name__ == "__main__":
    main()
