from pathlib import Path

from src.model_compat import load_compatible_model


MODEL_PATH = Path(__file__).resolve().parent / "models" / "bod_model.h5"
model = load_compatible_model(MODEL_PATH)
