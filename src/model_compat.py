from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import h5py
import tensorflow as tf
from tensorflow.keras.models import load_model

from src.cnn_lstm_attention import build_cnn_lstm_attention_model


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


def compat_custom_objects() -> Dict[str, object]:
    policy_cls = tf.keras.mixed_precision.Policy
    return {
        "GetItem": GetItem,
        "InputLayer": CompatInputLayer,
        "DTypePolicy": policy_cls,
        "Policy": policy_cls,
    }


def _extract_input_shape_from_h5(model_path: Path) -> Tuple[int, int]:
    with h5py.File(model_path, "r") as handle:
        raw_config = handle.attrs.get("model_config")
        if raw_config is None:
            raise ValueError(f"Model file does not contain a serialized config: {model_path}")

        if isinstance(raw_config, bytes):
            raw_config = raw_config.decode("utf-8")
        config = json.loads(raw_config)

    layers = config.get("config", {}).get("layers", [])
    for layer in layers:
        if layer.get("class_name") != "InputLayer":
            continue
        layer_config = layer.get("config", {})
        batch_shape = layer_config.get("batch_input_shape") or layer_config.get("batch_shape")
        if batch_shape and len(batch_shape) >= 3:
            time_steps = int(batch_shape[1])
            n_features = int(batch_shape[2])
            return time_steps, n_features

    raise ValueError(f"Unable to infer input shape from saved model config: {model_path}")


def _should_rebuild_from_weights(exc: Exception, model_path: Path) -> bool:
    if model_path.suffix.lower() not in {".h5", ".hdf5"}:
        return False

    message = str(exc)
    known_signals = (
        "quantization_config",
        "Keyword argument not understood",
        "Unknown layer",
        "GetItem",
        "InputLayer",
        "batch_shape",
        "optional",
        "Unknown dtype policy",
        "DTypePolicy",
        "Could not deserialize",
    )
    return any(signal in message for signal in known_signals)


def load_compatible_model(model_path: str | Path):
    model_path = Path(model_path)
    custom_objects = compat_custom_objects()

    try:
        model = load_model(model_path, custom_objects=custom_objects, compile=False)
    except Exception as exc:
        if not _should_rebuild_from_weights(exc, model_path):
            raise

        input_shape = _extract_input_shape_from_h5(model_path)
        model = build_cnn_lstm_attention_model(input_shape)
        model.load_weights(model_path)

    try:
        model.compile(optimizer="adam", loss="mse")
    except Exception:
        pass

    return model
