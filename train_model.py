import os
import cv2
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ============================
# CONFIG
# ============================

DATASET_RAW = "dataset_raw"      # chứa ảnh chưa segmentation
DATASET_SEG = "dataset_segmented"  # chứa ảnh sau segmentation
MODEL_PATH = "model/model.h5"
CLASS_MAP_PATH = "model/class_map.json"

IMG_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 20


# ============================
# 1. SEGMENTATION BẰNG OPENCV
# ============================

def segment_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold tách vật thể
    _, th = cv2.threshold(blur, 0, 255,
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=3)

    # Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    # Chọn contour lớn nhất → món bánh chính
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    crop = img[y:y+h, x:x+w]
    return crop


def build_segmented_dataset():
    print("🔧 Creating segmented dataset...")

    if not os.path.exists(DATASET_SEG):
        os.makedirs(DATASET_SEG)

    for class_name in os.listdir(DATASET_RAW):
        class_dir_raw = os.path.join(DATASET_RAW, class_name)
        class_dir_seg = os.path.join(DATASET_SEG, class_name)
        os.makedirs(class_dir_seg, exist_ok=True)

        for file in os.listdir(class_dir_raw):
            path = os.path.join(class_dir_raw, file)

            try:
                cropped = segment_image(path)
                if cropped is None:
                    continue

                cropped = cv2.resize(cropped, (IMG_SIZE, IMG_SIZE))
                cv2.imwrite(os.path.join(class_dir_seg, file), cropped)
            except:
                continue

    print("✅ Segmentation done.")


# ============================
# 2. CNN MODEL (TRAIN TỪ ĐẦU)
# ============================

def build_cnn(num_classes):
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

        layers.Conv2D(32, (3, 3), activation='relu', padding="same"),
        layers.MaxPooling2D(),

        layers.Conv2D(64, (3, 3), activation='relu', padding="same"),
        layers.MaxPooling2D(),

        layers.Conv2D(128, (3, 3), activation='relu', padding="same"),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),

        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================
# 3. TRAINING
# ============================

def train_model():
    build_segmented_dataset()

    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=15,
        zoom_range=0.1,
        horizontal_flip=True
    )

    train_gen = datagen.flow_from_directory(
        DATASET_SEG,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training"
    )

    val_gen = datagen.flow_from_directory(
        DATASET_SEG,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation"
    )

    num_classes = len(train_gen.class_indices)

    # Save class index map
    with open(CLASS_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(train_gen.class_indices, f, ensure_ascii=False, indent=2)

    model = build_cnn(num_classes)

    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen
    )

    model.save(MODEL_PATH)
    print("✅ Model saved at", MODEL_PATH)

    # Plot Loss + Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.title("Loss Curve")
    plt.legend()
    plt.savefig("logs/training_loss.png")

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="val")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.savefig("logs/training_accuracy.png")

    print("✅ Training plots saved.")


if __name__ == "__main__":
    train_model()
