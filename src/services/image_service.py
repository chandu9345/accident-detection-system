import os
import numpy as np
import tensorflow as tf
from PIL import Image
from src.common.config import get_paths, get_model_map
 
# Correct label map — must match training order
IMAGE_LABELS = {0: "Accident", 1: "Non-Accident"}
 
 
def _load_image(path, img_size=(224, 224)):
    img = Image.open(path).convert("RGB")
    img = img.resize(img_size)
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)
 
 
def predict_image(path: str) -> dict:
    paths = get_paths()
    model_map = get_model_map(paths["models_root"]) or {}
    model_path = os.path.join(
        paths["models_root"],
        model_map.get("image", {}).get("path", "image_model.h5")
    )
    model = tf.keras.models.load_model(model_path)
    x = _load_image(path)
    preds = model.predict(x)
 
    # Get both class scores
    accident_score     = float(preds[0][0])   # class 0 = Accident
    non_accident_score = float(preds[0][1])   # class 1 = Non-Accident
 
    cls_idx = int(np.argmax(preds, axis=1)[0])
    score   = float(np.max(preds))
    label   = IMAGE_LABELS[cls_idx]
 
    # ── Debug info (you can remove after confirming it works) ──
    print(f"[image_service] Raw predictions : {preds}")
    print(f"[image_service] Accident score  : {accident_score:.4f}")
    print(f"[image_service] Non-Accident    : {non_accident_score:.4f}")
    print(f"[image_service] Predicted class : {cls_idx} → {label}")
 
    return {
        "class_index": cls_idx,
        "score":       score,
        "label":       label,
        "all_scores": {
            "Accident":     round(accident_score, 4),
            "Non-Accident": round(non_accident_score, 4),
        }
    }
