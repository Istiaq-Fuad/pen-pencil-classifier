import tensorflow as tf
import numpy as np
import os
from pathlib import Path
import shutil
from tqdm import tqdm

tf.random.set_seed(42)
np.random.seed(42)


DATASET_PATH = Path("dataset")
OUTPUT_PATH = Path("processed_dataset")


IMG_HEIGHT = 224
IMG_WIDTH = 224
TARGET_COUNT = 500


def collect_images_by_class():
    pen_images = []
    pencil_images = []

    for root, dirs, files in os.walk(DATASET_PATH):
        parent_dir = Path(root).name.lower()

        if parent_dir == "pen":
            for file in files:
                if file.lower().endswith(".jpg"):
                    pen_images.append(os.path.join(root, file))
        elif parent_dir == "pencil":
            for file in files:
                if file.lower().endswith(".jpg"):
                    pencil_images.append(os.path.join(root, file))

    print(f"Found {len(pen_images)} pen images")
    print(f"Found {len(pencil_images)} pencil images")

    return pen_images, pencil_images


def augment_image(image):

    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)

    if tf.random.uniform(()) > 0.7:
        image = tf.image.flip_up_down(image)

    k = tf.random.uniform((), 0, 4, dtype=tf.int32)
    image = tf.image.rot90(image, k)

    if tf.random.uniform(()) > 0.5:

        zoom_factor = tf.random.uniform((), 0.8, 0.95)
        h, w = IMG_HEIGHT, IMG_WIDTH
        crop_h = tf.cast(tf.cast(h, tf.float32) * zoom_factor, tf.int32)
        crop_w = tf.cast(tf.cast(w, tf.float32) * zoom_factor, tf.int32)
        image = tf.image.random_crop(image, [crop_h, crop_w, 3])
        image = tf.image.resize(image, [h, w])

    image = tf.image.random_brightness(image, max_delta=40)

    image = tf.image.random_contrast(image, lower=0.6, upper=1.4)

    image = tf.image.random_saturation(image, lower=0.6, upper=1.4)

    image = tf.image.random_hue(image, max_delta=0.15)

    if tf.random.uniform(()) > 0.5:
        quality = tf.random.uniform((), 75, 100, dtype=tf.int32)

        image = tf.cast(image, tf.uint8)
        encoded = tf.io.encode_jpeg(image, quality=quality)
        image = tf.io.decode_jpeg(encoded, channels=3)
        image = tf.cast(image, tf.float32)

    image = tf.clip_by_value(image, 0.0, 255.0)

    return image


def load_and_resize_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    return img


def create_augmented_dataset_on_disk():
    pen_images, pencil_images = collect_images_by_class()

    if len(pen_images) == 0 or len(pencil_images) == 0:
        raise ValueError("No images found! Please check the dataset path.")

    pen_dir = OUTPUT_PATH / "pen"
    pencil_dir = OUTPUT_PATH / "pencil"

    pen_dir.mkdir(parents=True, exist_ok=True)
    pencil_dir.mkdir(parents=True, exist_ok=True)

    for class_name, image_paths, output_dir in [
        ("pen", pen_images, pen_dir),
        ("pencil", pencil_images, pencil_dir),
    ]:
        current_count = len(image_paths)
        augmentations_needed = TARGET_COUNT - current_count

        np.random.shuffle(image_paths)

        all_images = []

        for img_path in tqdm(image_paths, desc="Loading"):
            try:
                img = load_and_resize_image(img_path)
                all_images.append(img.numpy())
            except Exception as e:
                print(f"\nWarning: Could not load {img_path}: {e}")
                continue

        if augmentations_needed > 0:
            for i in tqdm(range(augmentations_needed), desc="Augmenting"):
                source_img = all_images[i % len(all_images)]

                aug_img = augment_image(source_img)
                all_images.append(aug_img.numpy())

        np.random.shuffle(all_images)

        all_images = all_images[:TARGET_COUNT]

        for idx, img in enumerate(tqdm(all_images, desc=f"Saving {class_name}")):
            img_path = output_dir / f"{class_name}_{idx:04d}.jpg"
            tf.keras.preprocessing.image.save_img(img_path, img)

    return OUTPUT_PATH


if __name__ == "__main__":
    print("Creating Augmented Pen/Pencil Dataset")

    pen_images, pencil_images = collect_images_by_class()

    output_path = create_augmented_dataset_on_disk()
