import tensorflow as tf
import os
from pathlib import Path


DATASET_PATH = Path("processed_dataset")


IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32


def count_images():
    pen_dir = DATASET_PATH / "pen"
    pencil_dir = DATASET_PATH / "pencil"

    pen_count = len(list(pen_dir.glob("*.jpg"))) if pen_dir.exists() else 0
    pencil_count = len(list(pencil_dir.glob("*.jpg"))) if pencil_dir.exists() else 0

    return pen_count, pencil_count


def load_dataset(validation_split=0.2, seed=42):

    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH,
        validation_split=validation_split,
        subset="training",
        seed=seed,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        label_mode="binary",
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        label_mode="binary",
    )

    class_names = train_ds.class_names

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names


def load_full_dataset(shuffle=True, seed=42):

    dataset = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH,
        seed=seed,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        label_mode="binary",
    )

    class_names = dataset.class_names

    AUTOTUNE = tf.data.AUTOTUNE
    dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)

    print(f"Total batches: {len(dataset)}")

    return dataset, class_names


if __name__ == "__main__":

    train_ds, val_ds, class_names = load_dataset(validation_split=0.2)
