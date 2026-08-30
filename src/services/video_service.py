import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os

MODEL_PATH = "models/image_model.h5"
IMG_SIZE   = (224, 224)
CLASSES    = ["Accident", "Non Accident"]
NUM_FRAMES = 16

_model = None

def _get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        _model = load_model(MODEL_PATH)
    return _model

def predict_video(video_path: str) -> dict:
    model = _get_model()

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        cap.release()
        raise ValueError("Could not read video file.")

    indices = np.linspace(0, total_frames - 1, NUM_FRAMES, dtype=int)

    scores = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        resized   = cv2.resize(frame, IMG_SIZE)
        img_array = img_to_array(resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        pred = model.predict(img_array, verbose=0)
        scores.append(pred[0])

    cap.release()

    if not scores:
        raise ValueError("No frames could be extracted from video.")

    avg_scores  = np.mean(scores, axis=0)
    class_index = int(np.argmax(avg_scores))
    confidence  = float(avg_scores[class_index])

    return {
        "class_index" : class_index,
        "score"       : confidence,
        "label"       : CLASSES[class_index],
        "frames_used" : len(scores),
    }