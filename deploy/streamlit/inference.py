from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import json

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model


class GetItem(tf.keras.layers.Layer):
    """Compatibility layer for models saved with tensor getitem slicing."""

    def __init__(self, s=None, index=None, **kwargs):
        super().__init__(**kwargs)
        self.s = s if s is not None else index

    def __call__(self, inputs, *args, **kwargs):
        if args:
            kwargs.setdefault("s", args[0])
        return super().__call__(inputs, **kwargs)

    def call(self, inputs, s=None):
        s = self.s if s is None else s
        try:
            return inputs[s]
        except Exception:
            if isinstance(s, int):
                return inputs[:, s, :]
            if isinstance(s, (list, tuple)) and len(s) >= 2:
                idx = s[1]
                if isinstance(idx, int):
                    return inputs[:, idx, :]
            raise

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"s": self.s})
        return cfg


class CompatInputLayer(tf.keras.layers.InputLayer):
    """Compatibility InputLayer for older H5 configs across keras versions."""

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        if "batch_shape" in config and "batch_input_shape" not in config:
            config["batch_input_shape"] = config.pop("batch_shape")
        config.pop("optional", None)
        return super().from_config(config)


def _compat_custom_objects() -> Dict[str, object]:
    # Keras version differences can serialize dtype policy as DTypePolicy or Policy.
    policy_cls = tf.keras.mixed_precision.Policy
    return {
        "GetItem": GetItem,
        "InputLayer": CompatInputLayer,
        "DTypePolicy": policy_cls,
        "Policy": policy_cls,
    }


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _normalize_target(target_name: str) -> str:
    return (
        target_name.strip()
        .lower()
        .replace("(mg/l)", "")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
        .replace(" ", "_")
        .strip("_")
    )


TARGET_MODEL_CANDIDATES: Dict[str, List[Path]] = {
    "BOD (mg/L)": [
        Path("models/cnn_lstm_attention_model.h5"),
        Path("models/bod_cnn_lstm_attention_model.h5"),
        Path("smart_stp_predictor/backend/models/bod_model.h5"),
    ],
    "COD (mg/L)": [
        Path("models/cod_cnn_lstm_attention_model.h5"),
    ],
    "PH": [
        Path("models/ph_cnn_lstm_attention_model.h5"),
    ],
    "DO (mg/L)": [
        Path("models/do_cnn_lstm_attention_model.h5"),
    ],
}

TARGET_ARTIFACT_PREFIX: Dict[str, str] = {
    "BOD (mg/L)": "",
    "COD (mg/L)": "cod",
    "PH": "ph",
    "DO (mg/L)": "do",
}


def available_targets() -> List[str]:
    root = _project_root()
    available = []
    for target, candidates in TARGET_MODEL_CANDIDATES.items():
        if any((root / candidate).exists() for candidate in candidates):
            available.append(target)
    return available


def resolve_artifact_paths(target: str) -> Dict[str, Optional[Path]]:
    root = _project_root()
    prefix = TARGET_ARTIFACT_PREFIX.get(target, _normalize_target(target))

    if prefix:
        report_dir = root / "reports"
        plot_dir = root / "plots"
        return {
            "metrics": report_dir / f"{prefix}_metrics.json",
            "report": report_dir / f"{prefix}_detailed_report.txt",
            "training_history": plot_dir / f"{prefix}_01_training_history.png",
            "predictions": plot_dir / f"{prefix}_02_predictions_vs_actual.png",
            "residuals": plot_dir / f"{prefix}_03_residuals.png",
            "metrics_plot": plot_dir / f"{prefix}_04_metrics_summary.png",
            "scatter": plot_dir / f"{prefix}_05_actual_vs_predicted_scatter.png",
        }

    return {
        "metrics": root / "reports" / "metrics.json",
        "report": root / "reports" / "detailed_report.txt",
        "training_history": root / "plots" / "01_training_history.png",
        "predictions": root / "plots" / "02_predictions_vs_actual.png",
        "residuals": root / "plots" / "03_residual_plot.png",
        "metrics_plot": root / "plots" / "04_metrics_summary.png",
        "scatter": root / "plots" / "05_actual_vs_predicted_scatter.png",
    }


def load_metrics(target: str) -> Tuple[Optional[Dict[str, float]], Optional[Path]]:
    artifact_paths = resolve_artifact_paths(target)
    metrics_path = artifact_paths["metrics"]
    if metrics_path is None or not metrics_path.exists():
        return None, None

    with metrics_path.open("r", encoding="utf-8") as file_handle:
        raw_metrics = json.load(file_handle)

    metrics: Dict[str, float] = {}
    for key, value in raw_metrics.items():
        normalized_key = key.replace("R²", "R2").replace("R�", "R2").replace("R2 Score", "R2 Score")
        if "r" in normalized_key.lower() and "score" in normalized_key.lower():
            normalized_key = "R2 Score"
        metrics[normalized_key] = value

    return metrics, metrics_path


def load_report_text(target: str) -> Tuple[Optional[str], Optional[Path]]:
    artifact_paths = resolve_artifact_paths(target)
    report_path = artifact_paths["report"]
    if report_path is None or not report_path.exists():
        return None, None

    return report_path.read_text(encoding="utf-8"), report_path


def resolve_model_path(target: str) -> Path:
    root = _project_root()

    if target in TARGET_MODEL_CANDIDATES:
        for candidate in TARGET_MODEL_CANDIDATES[target]:
            full_path = root / candidate
            if full_path.exists():
                return full_path

    slug = _normalize_target(target)
    fallback = root / "models" / f"{slug}_cnn_lstm_attention_model.h5"
    if fallback.exists():
        return fallback

    raise FileNotFoundError(f"No model file found for target '{target}'.")


def load_prediction_model(target: str):
    model_path = resolve_model_path(target)
    custom_objects = _compat_custom_objects()
    try:
        model = load_model(model_path, custom_objects=custom_objects, compile=False)
    except Exception as exc:
        message = str(exc)
        if (
            "GetItem" in message
            or "Unknown layer" in message
            or "InputLayer" in message
            or "batch_shape" in message
            or "optional" in message
            or "Unknown dtype policy" in message
            or "DTypePolicy" in message
        ):
            model = load_model(
                model_path,
                custom_objects=custom_objects,
                compile=False,
            )
        else:
            raise

    try:
        model.compile(optimizer="adam", loss="mse")
    except Exception:
        pass

    return model, model_path


def get_expected_feature_count(model) -> int:
    input_shape = getattr(model, "input_shape", None)
    if not input_shape or len(input_shape) < 3 or input_shape[-1] is None:
        raise ValueError("Unable to determine the model's expected feature count.")
    return int(input_shape[-1])


def select_feature_columns(numeric_cols: List[str], target_col: str, expected_features: int) -> List[str]:
    if target_col not in numeric_cols:
        raise ValueError(f"Target column '{target_col}' must be numeric to run predictions.")

    if len(numeric_cols) < expected_features:
        raise ValueError(
            f"The uploaded dataset has {len(numeric_cols)} numeric columns, but the model expects {expected_features}."
        )

    selected_cols = numeric_cols[:expected_features]
    if target_col not in selected_cols:
        raise ValueError(
            f"Target column '{target_col}' is not included in the first {expected_features} numeric columns expected by the model."
        )

    return selected_cols


def build_residual_analysis(
    data: pd.DataFrame,
    target_name: str,
    time_steps: int = 30,
    max_points: int = 200,
) -> Tuple[pd.DataFrame, Dict[str, float], str]:
    if "Date" in data.columns:
        data = data.copy()
        data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
        data = data.sort_values("Date")
        data = data.set_index("Date")

    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric columns found in uploaded dataset.")

    target_col = find_target_column(data[numeric_cols], target_name)
    model, model_path = load_prediction_model(target_name)
    expected_features = get_expected_feature_count(model)
    selected_cols = select_feature_columns(numeric_cols, target_col, expected_features)
    target_index = selected_cols.index(target_col)

    df_num = data[selected_cols].astype(float)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_num.values)

    if scaled.shape[0] <= time_steps:
        raise ValueError(
            f"Not enough rows for residual analysis. Required more than {time_steps}, got {scaled.shape[0]}."
        )

    start_index = max(time_steps, scaled.shape[0] - max_points)

    rows = []
    for end_index in range(start_index, scaled.shape[0]):
        sequence = scaled[end_index - time_steps:end_index]
        x = sequence.reshape(1, time_steps, len(selected_cols))
        pred_scaled = model.predict(x, verbose=0)
        pred_scaled_value = float(np.array(pred_scaled).ravel()[0])

        dummy = np.zeros((1, len(selected_cols)), dtype=float)
        dummy[0, target_index] = pred_scaled_value
        pred_original = float(scaler.inverse_transform(dummy)[0, target_index])
        actual_original = float(df_num.iloc[end_index][target_col])
        residual = actual_original - pred_original

        rows.append(
            {
                "Index": end_index,
                "Actual": actual_original,
                "Predicted": pred_original,
                "Residual": residual,
            }
        )

    residual_df = pd.DataFrame(rows)
    rmse = float(np.sqrt(np.mean(np.square(residual_df["Residual"]))))
    mae = float(np.mean(np.abs(residual_df["Residual"])))
    actual_values = residual_df["Actual"].to_numpy()
    predicted_values = residual_df["Predicted"].to_numpy()
    ss_res = float(np.sum((actual_values - predicted_values) ** 2))
    ss_tot = float(np.sum((actual_values - np.mean(actual_values)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot if ss_tot else 0.0)
    mape = float(np.mean(np.abs((actual_values - predicted_values) / np.maximum(np.abs(actual_values), 1e-8))) * 100)

    metrics = {
        "RMSE": rmse,
        "MAE": mae,
        "R2 Score": r2,
        "MAPE (%)": mape,
    }
    return residual_df, metrics, str(model_path)


def find_target_column(df: pd.DataFrame, target_name: str) -> str:
    if target_name in df.columns:
        return target_name

    target_lower = target_name.strip().lower()
    for col in df.columns:
        col_lower = col.strip().lower()
        if col_lower == target_lower:
            return col
        if target_lower in col_lower or col_lower in target_lower:
            return col

    normalized_aliases = {
        "bod": ["bod"],
        "cod": ["cod"],
        "ph": ["ph", "ph value", "power of hydrogen"],
        "do": ["do", "dissolved oxygen"],
    }

    for col in df.columns:
        col_lower = col.lower()
        for canonical, aliases in normalized_aliases.items():
            if target_lower in aliases and canonical in col_lower:
                return col

    raise ValueError(
        f"Target column '{target_name}' not found in uploaded data columns: {list(df.columns)}"
    )


def predict_future(
    data: pd.DataFrame,
    target_name: str,
    days: int = 7,
    time_steps: int = 30,
) -> Tuple[pd.DataFrame, str]:
    if "Date" in data.columns:
        data = data.copy()
        data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
        data = data.sort_values("Date")
        data = data.set_index("Date")

    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric columns found in uploaded dataset.")

    target_col = find_target_column(data[numeric_cols], target_name)

    model, model_path = load_prediction_model(target_name)
    expected_features = get_expected_feature_count(model)
    selected_cols = select_feature_columns(numeric_cols, target_col, expected_features)
    target_index = selected_cols.index(target_col)

    df_num = data[selected_cols].astype(float)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_num.values)

    if scaled.shape[0] < time_steps:
        raise ValueError(
            f"Not enough rows for prediction. Required at least {time_steps}, got {scaled.shape[0]}."
        )

    last_seq = scaled[-time_steps:].copy()
    current_sequence = last_seq.copy()
    n_features = scaled.shape[1]

    predictions = []
    for day in range(1, days + 1):
        x = current_sequence.reshape(1, time_steps, n_features)
        pred = model.predict(x, verbose=0)
        pred_val = float(np.array(pred).ravel()[0])

        dummy = np.zeros((1, n_features), dtype=float)
        dummy[0, target_index] = pred_val
        inv = scaler.inverse_transform(dummy)
        predicted_value = float(inv[0, target_index])

        predictions.append({"Day": day, f"Predicted {target_col}": predicted_value})

        new_row = current_sequence[-1].copy()
        new_row[target_index] = pred_val
        current_sequence = np.vstack([current_sequence[1:], new_row])

    result_df = pd.DataFrame(predictions)
    return result_df, str(model_path)
