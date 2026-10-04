"""
inference_fix.py  --  drop-in inference code for the Streamlit app.

Put this file next to app.py in your GitHub repo:

    your-repo/
    ├── app.py
    ├── inference_fix.py          <-- this file
    ├── requirements.txt
    └── models/
        ├── efficientnet_v2_final.keras
        ├── dnn_v2_final.keras
        └── hybrid_config_v2.json

What it fixes
-------------
1. CONFIG BUG: the notebook saves  {"efficientnet_weight", "dnn_weight", "threshold"}
   but the old app looked for "BEST_EFF_WEIGHT"/"best_eff_weight"/"eff_weight",
   never found it, and silently fell back to 0.5.  The app was therefore computing
       hybrid = 0.5 * eff + 0.8 * dnn      (weights sum to 1.3, not 1.0)
   Here the keys are read strictly and the weights must sum to 1.
2. PATHS: relative to this file instead of /kaggle/working/...
3. PREPROCESSING: one function, with the resize mode explicit (see diagnose_pipeline.py).
4. UNCERTAIN verdict when the two heads strongly disagree or the score sits on the threshold.
"""

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
EFF_PATH = MODEL_DIR / "efficientnet_v2_final.keras"
DNN_PATH = MODEL_DIR / "dnn_v2_final.keras"
CONFIG_PATH = MODEL_DIR / "hybrid_config_v2.json"

IMG_SIZE = 256

# "tf_bilinear" == what the training pipeline did (tf.image.resize, no antialias).
# Change this only after diagnose_pipeline.py shows another mode behaves better.
RESIZE_MODE = "tf_bilinear"

# Tune these on validation data, not by feel.
DISAGREEMENT_LIMIT = 0.50   # |eff - dnn| above this -> UNCERTAIN
BORDERLINE_MARGIN = 0.05    # |hybrid - threshold| below this -> UNCERTAIN


def load_config(path=CONFIG_PATH):
    """Read weights + threshold. Raises instead of silently using defaults."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    required = ("efficientnet_weight", "dnn_weight", "threshold")
    missing = [k for k in required if k not in cfg]
    if missing:
        raise KeyError(f"{path} is missing keys {missing}; found {list(cfg)}")

    eff_w = float(cfg["efficientnet_weight"])
    dnn_w = float(cfg["dnn_weight"])
    thr = float(cfg["threshold"])

    if abs(eff_w + dnn_w - 1.0) > 1e-6:
        raise ValueError(f"Fusion weights must sum to 1, got {eff_w} + {dnn_w}")
    return eff_w, dnn_w, thr


def load_models():
    """Returns (eff_model, dnn_model, feature_model, eff_w, dnn_w, threshold).
    Wrap this with @st.cache_resource in app.py."""
    eff_model = tf.keras.models.load_model(EFF_PATH, compile=False)
    dnn_model = tf.keras.models.load_model(DNN_PATH, compile=False)

    feature_model = None
    for layer in reversed(eff_model.layers[:-1]):
        shape = tuple(layer.output.shape)
        if len(shape) == 2 and shape[-1] == 1280:
            feature_model = tf.keras.Model(eff_model.input, layer.output)
            break
    if feature_model is None:
        raise RuntimeError("Could not find the 1280-d feature layer in the EfficientNet model.")

    eff_w, dnn_w, thr = load_config()
    return eff_model, dnn_model, feature_model, eff_w, dnn_w, thr


def preprocess(image: Image.Image, mode: str = RESIZE_MODE) -> np.ndarray:
    """PIL image -> float32 array of shape (1, 256, 256, 3), pixel range 0-255.
    EfficientNet in Keras rescales internally, so do NOT divide by 255 here."""
    image = image.convert("RGB")

    if mode == "pil_bilinear":
        arr = np.asarray(image.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR), dtype=np.float32)
    elif mode == "pil_lanczos":
        arr = np.asarray(image.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS), dtype=np.float32)
    elif mode in ("tf_bilinear", "tf_antialias"):
        raw = np.asarray(image, dtype=np.float32)
        arr = tf.image.resize(
            raw, [IMG_SIZE, IMG_SIZE], method="bilinear", antialias=(mode == "tf_antialias")
        ).numpy()
    else:
        raise ValueError(f"Unknown resize mode: {mode}")

    return arr[None, ...].astype(np.float32)


def predict(image, eff_model, dnn_model, feature_model, eff_w, dnn_w, threshold, mode=RESIZE_MODE):
    batch = preprocess(image, mode)

    eff_p = float(eff_model(batch, training=False).numpy().reshape(-1)[0])
    feats = feature_model(batch, training=False)
    dnn_p = float(dnn_model(feats, training=False).numpy().reshape(-1)[0])

    hybrid = eff_w * eff_p + dnn_w * dnn_p
    label = "AI-GENERATED" if hybrid >= threshold else "REAL"

    disagreement = abs(eff_p - dnn_p)
    margin = abs(hybrid - threshold)
    if disagreement > DISAGREEMENT_LIMIT or margin < BORDERLINE_MARGIN:
        verdict = "UNCERTAIN"
    else:
        verdict = label

    return {
        "verdict": verdict,                 # show this in the UI
        "label": label,                     # raw thresholded label
        "hybrid_ai_score": hybrid,          # always in [0, 1] now
        "efficientnet_score": eff_p,
        "dnn_score": dnn_p,
        "threshold": threshold,
        "margin": margin,
        "disagreement": disagreement,
    }
