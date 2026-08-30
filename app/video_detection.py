import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# ── Config ──────────────────────────────────────
MODEL_PATH = "models/image_model.h5"
IMG_SIZE   = (224, 224)
CLASSES    = ["Accident", "Non Accident"]
THRESHOLD  = 0.7
# ────────────────────────────────────────────────

# Load model
model = load_model(MODEL_PATH)
print("✅ Model loaded!")

# 0 = webcam | or give video path like "myvideo.mp4"
cap = cv2.VideoCapture("myvideo.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess
    resized   = cv2.resize(frame, IMG_SIZE)
    img_array = img_to_array(resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array, verbose=0)
    class_idx  = np.argmax(prediction[0])
    confidence = prediction[0][class_idx]
    label      = CLASSES[class_idx]

    # Display result on frame
    if label == "Accident" and confidence >= THRESHOLD:
        color = (0, 0, 255)  # Red
        text  = f"ACCIDENT DETECTED! {confidence*100:.1f}%"
        cv2.rectangle(frame, (10, 10),
                      (frame.shape[1]-10, frame.shape[0]-10), color, 4)
    else:
        color = (0, 255, 0)  # Green
        text  = f"No Accident {confidence*100:.1f}%"

    cv2.putText(frame, text, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    cv2.imshow("Accident Detection", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()