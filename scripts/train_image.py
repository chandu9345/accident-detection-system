"""
scripts/train_image.py
 
Improved training script with:
  - MobileNetV2 transfer learning (pretrained on ImageNet)
  - Heavy data augmentation to compensate for small dataset
  - Class weights to handle imbalance
  - Learning rate scheduler
  - Early stopping to prevent overfitting
 
Run:
    python scripts/train_image.py
"""
 
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
)
 
# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAIN_DIR  = os.path.join(BASE_DIR, "data", "accident_dataset", "train")
VAL_DIR    = os.path.join(BASE_DIR, "data", "accident_dataset", "val")
MODEL_OUT  = os.path.join(BASE_DIR, "models", "image_model.h5")
 
IMG_SIZE   = (224, 224)
BATCH_SIZE = 16        # small batch — fine for 791 images
EPOCHS     = 50        # early stopping will kick in before this
LR         = 1e-4
 
# ── Data Augmentation (increases effective dataset size) ──────────────────────
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.10,
    zoom_range=0.20,
    brightness_range=[0.7, 1.3],
    horizontal_flip=True,
    fill_mode="nearest",
)
 
val_datagen = ImageDataGenerator(rescale=1.0 / 255)
 
train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True,
    seed=42,
)
 
val_gen = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
)
 
print("\n✅ Class indices:", train_gen.class_indices)
# Should print: {'Accident': 0, 'Non_Accident': 1}
 
# ── Class Weights (handles imbalance) ────────────────────────────────────────
total   = sum(train_gen.samples for _ in [1])
n_acc   = train_gen.classes.tolist().count(0)
n_nonacc= train_gen.classes.tolist().count(1)
 
class_weight = {
    0: total / (2 * n_acc),
    1: total / (2 * n_nonacc),
}
print(f"⚖️  Class weights: {class_weight}")
 
# ── Model — MobileNetV2 + custom head ────────────────────────────────────────
base = MobileNetV2(
    input_shape=(*IMG_SIZE, 3),
    include_top=False,
    weights="imagenet",
)
 
# Phase 1: freeze base, train only head
base.trainable = False
 
x = base.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.3)(x)
out = layers.Dense(2, activation="softmax")(x)
 
model = Model(inputs=base.input, outputs=out)
 
model.compile(
    optimizer=tf.keras.optimizers.Adam(LR),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
 
print(f"\n🚀 Phase 1 — Training head only ({EPOCHS//2} epochs)...")
 
callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=8,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                      patience=4, min_lr=1e-7, verbose=1),
    ModelCheckpoint(MODEL_OUT, monitor="val_accuracy",
                    save_best_only=True, verbose=1),
]
 
model.fit(
    train_gen,
    epochs=EPOCHS // 2,
    validation_data=val_gen,
    class_weight=class_weight,
    callbacks=callbacks,
)
 
# Phase 2: unfreeze last 30 layers for fine-tuning
print("\n🔥 Phase 2 — Fine-tuning last 30 layers...")
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False
 
model.compile(
    optimizer=tf.keras.optimizers.Adam(LR / 10),  # lower LR for fine-tuning
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
 
model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    class_weight=class_weight,
    callbacks=callbacks,
)
 
print(f"\n✅ Model saved to: {MODEL_OUT}")
 
# ── Final Evaluation ──────────────────────────────────────────────────────────
print("\n📊 Final Evaluation on Validation Set:")
loss, acc = model.evaluate(val_gen, verbose=0)
print(f"   Accuracy : {acc*100:.2f}%")
print(f"   Loss     : {loss:.4f}")
