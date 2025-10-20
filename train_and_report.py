import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path
from load_processed_dataset import load_dataset, load_full_dataset

tf.random.set_seed(42)
np.random.seed(42)

OUTPUT_DIR = Path("output_images")
OUTPUT_DIR.mkdir(exist_ok=True)
EPOCHS = 10


def create_test_split(full_dataset, test_split=0.1):
    total_batches = len(full_dataset)
    test_size = int(total_batches * test_split)
    train_val_size = total_batches - test_size

    train_val_ds = full_dataset.take(train_val_size)
    test_ds = full_dataset.skip(train_val_size)

    return train_val_ds, test_ds


def create_cnn_model(input_shape=(224, 224, 3)):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Rescaling(1.0 / 255, input_shape=input_shape),
            tf.keras.layers.Conv2D(32, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(64, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(128, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="PenPencilCNN",
    )

    return model


def count_parameters(model):
    trainable = np.sum([np.prod(v.shape) for v in model.trainable_weights])
    non_trainable = np.sum([np.prod(v.shape) for v in model.non_trainable_weights])
    return int(trainable), int(non_trainable)


def plot_accuracy_loss(history, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history.history["accuracy"]) + 1)

    ax1.plot(epochs, history.history["accuracy"], "b-o", label="Training", markersize=5)
    ax1.plot(
        epochs,
        history.history["val_accuracy"],
        "r--s",
        label="Validation",
        markersize=5,
    )
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.set_title("Model Accuracy", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Loss
    ax2.plot(epochs, history.history["loss"], "b-o", label="Training", markersize=5)
    ax2.plot(
        epochs, history.history["val_loss"], "r--s", label="Validation", markersize=5
    )
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Loss", fontsize=12)
    ax2.set_title("Model Loss", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_metrics(history, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history.history["precision"]) + 1)

    ax1.plot(
        epochs,
        history.history["precision"],
        "g-o",
        label="Train Precision",
        markersize=5,
    )
    ax1.plot(
        epochs,
        history.history["val_precision"],
        "g--s",
        label="Val Precision",
        markersize=5,
    )
    ax1.plot(
        epochs, history.history["recall"], "m-o", label="Train Recall", markersize=5
    )
    ax1.plot(
        epochs, history.history["val_recall"], "m--s", label="Val Recall", markersize=5
    )
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Score", fontsize=12)
    ax1.set_title("Precision & Recall", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history.history["auc"], "b-o", label="Train AUC", markersize=5)
    ax2.plot(epochs, history.history["val_auc"], "r--s", label="Val AUC", markersize=5)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("AUC", fontsize=12)
    ax2.set_title("AUC Score", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_performance_bars(test_metrics, save_path):
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics_names = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC"]
    metrics_values = [
        test_metrics["accuracy"],
        test_metrics["precision"],
        test_metrics["recall"],
        test_metrics["f1_score"],
        test_metrics["auc"],
    ]

    colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6"]
    bars = ax.bar(
        metrics_names,
        metrics_values,
        color=colors,
        alpha=0.8,
        edgecolor="black",
        linewidth=1.5,
    )

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Test Set Performance Metrics", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    full_ds, class_names = load_full_dataset(shuffle=True, seed=42)
    print(f"   Classes: {class_names}")

    train_val_ds, test_ds = create_test_split(full_ds, test_split=0.1)

    train_ds, val_ds, _ = load_dataset(validation_split=0.2, seed=42)

    model = create_cnn_model()

    trainable_params, non_trainable_params = count_parameters(model)
    total_params = trainable_params + non_trainable_params

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    
    start_time = time.time()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        verbose=1,
    )

    training_time = time.time() - start_time
    print(
        f"\nTraining completed in {training_time:.2f}s ({training_time/60:.2f} min)"
    )

    test_results = model.evaluate(test_ds, verbose=1)

    test_metrics = {
        "loss": float(test_results[0]),
        "accuracy": float(test_results[1]),
        "precision": float(test_results[2]),
        "recall": float(test_results[3]),
        "auc": float(test_results[4]),
    }

    if test_metrics["precision"] + test_metrics["recall"] > 0:
        test_metrics["f1_score"] = (
            2
            * (test_metrics["precision"] * test_metrics["recall"])
            / (test_metrics["precision"] + test_metrics["recall"])
        )
    else:
        test_metrics["f1_score"] = 0.0

    model_path = "pen_pencil_cnn_model.keras"
    model.save(model_path)

    plot_accuracy_loss(history, OUTPUT_DIR / "accuracy_loss.png")
    plot_metrics(history, OUTPUT_DIR / "precision_recall_auc.png")
    plot_performance_bars(test_metrics, OUTPUT_DIR / "test_performance.png")

    print("Total Parameters:", total_params)

if __name__ == "__main__":
    main()
