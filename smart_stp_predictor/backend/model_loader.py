from tensorflow.keras.models import load_model
import tensorflow as tf

class GetItem(tf.keras.layers.Layer):
    def __init__(self, s=None, index=None, **kwargs):
        super().__init__(**kwargs)
        self.s = s if s is not None else index

    def __call__(self, inputs, *args, **kwargs):
        if args:
            kwargs.setdefault('s', args[0])
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

try:
    model = load_model("models/bod_model.h5", compile=False)
except ValueError as e:
    msg = str(e)
    if "GetItem" in msg or "Unknown layer" in msg:
        model = load_model("models/bod_model.h5", custom_objects={"GetItem": GetItem}, compile=False)
    else:
        raise
try:
    model.compile(optimizer="adam", loss="mse")
except Exception:
    pass