# src/services/severity_service.py
import os
import numpy as np
from PIL import Image

# ── Class labels ──────────────────────────────────────────────────────────
SEVERITY_CLASSES = ["Minor", "Moderate", "Severe"]

def predict_severity(path: str) -> dict:
    """
    Predict accident severity using the trained MobileNetV2 model.
    Returns: {"class_index": int, "score": float, "label": str}
    """
    # ── Find model ────────────────────────────────────────────────────────
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
    model_path = os.path.join(project_root, "models", "severity_model.h5")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Severity model weights not found at `{model_path}`. "
            "Please train the model first using "
            "`python -m scripts.train_image --data-dir data/severity_dataset "
            "--model-out models/severity_model.h5`"
        )

    # ── Load model ────────────────────────────────────────────────────────
    import tensorflow as tf
    model = tf.keras.models.load_model(model_path)

    # ── Preprocess image ──────────────────────────────────────────────────
    img = Image.open(path).convert("RGB").resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    # ── Predict ───────────────────────────────────────────────────────────
    predictions = model.predict(img_array, verbose=0)
    class_index = int(np.argmax(predictions[0]))
    score       = float(np.max(predictions[0]))
    label       = SEVERITY_CLASSES[class_index]

    return {
        "class_index": class_index,
        "score"      : score,
        "label"      : label,
    }