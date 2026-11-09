"""Train EfficientNetV2B0 for groundnut leaf disease classification."""
import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from src.model_utils import IMAGE_SIZE, CLASS_NAMES, SEED, build_model

AUTOTUNE = tf.data.AUTOTUNE


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="data/groundnut_leaf_dataset")
    parser.add_argument("--model_dir", default="models")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--fine_tune_epochs", type=int, default=10)
    parser.add_argument("--initial_lr", type=float, default=1e-3)
    parser.add_argument("--fine_tune_lr", type=float, default=1e-5)
    return parser.parse_args()


def main():
    args = parse_args()
    tf.keras.utils.set_random_seed(SEED)

    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    missing = [name for name in CLASS_NAMES if not (data_dir / name).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"Missing class folders in {data_dir}: {missing}\n"
            "See README.md for the required dataset layout."
        )

    common = dict(
        directory=str(data_dir),
        validation_split=0.2,
        seed=SEED,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=args.batch_size,
        label_mode="int",
        class_names=CLASS_NAMES,
    )
    train_ds = tf.keras.utils.image_dataset_from_directory(
        subset="training", shuffle=True, **common
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        subset="validation", shuffle=False, **common
    )

    print("Class order:", train_ds.class_names)
    print("Training images:", train_ds.cardinality().numpy(), "batches")
    # Avoid caching the full decoded image dataset in RAM; prefetch overlaps I/O and compute.
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)

    model = build_model(len(CLASS_NAMES))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.initial_lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=2, min_lr=1e-7
        ),
    ]

    print(f"\nStarting initial training ({args.epochs} epochs)")
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    # Fine-tune upper backbone layers, following the report's layer-100 strategy.
    backbone = next(
        layer for layer in model.layers
        if isinstance(layer, tf.keras.Model) and "efficientnet" in layer.name.lower()
    )
    backbone.trainable = True
    for layer in backbone.layers[:100]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.fine_tune_lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    print(f"\nStarting fine-tuning ({args.fine_tune_epochs} epochs)")
    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.fine_tune_epochs,
        callbacks=callbacks,
    )

    # Save the final model and the exact class-label mapping.
    model_path = model_dir / "groundnut_efficientnetv2b0.keras"
    model.save(model_path)
    with open(model_dir / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(CLASS_NAMES, f, indent=2)

    # Evaluate on validation data.
    y_true, y_prob = [], []
    for images, labels in val_ds:
        y_true.extend(labels.numpy().tolist())
        y_prob.extend(model.predict(images, verbose=0).tolist())
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = np.argmax(y_prob, axis=1)

    report = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, digits=4, zero_division=0
    )
    metrics_text = (
        f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}\n"
        f"Precision (weighted): {precision_score(y_true, y_pred, average='weighted', zero_division=0):.4f}\n"
        f"Recall (weighted):    {recall_score(y_true, y_pred, average='weighted', zero_division=0):.4f}\n"
        f"F1 (weighted):        {f1_score(y_true, y_pred, average='weighted', zero_division=0):.4f}\n\n"
        f"Class-wise report:\n{report}"
    )
    print("\n" + metrics_text)
    (output_dir / "classification_report.txt").write_text(metrics_text, encoding="utf-8")

    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES)))
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(
        ax=ax, xticks_rotation=35, cmap="Blues", colorbar=False
    )
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    # Combine the two training phases for plotting.
    combined = {}
    for key in ("accuracy", "val_accuracy", "loss", "val_loss"):
        combined[key] = history1.history.get(key, []) + history2.history.get(key, [])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(combined["accuracy"], label="Training")
    axes[0].plot(combined["val_accuracy"], label="Validation")
    axes[0].axvline(len(history1.history["accuracy"]) - 1, linestyle="--", label="Fine-tuning")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(combined["loss"], label="Training")
    axes[1].plot(combined["val_loss"], label="Validation")
    axes[1].axvline(len(history1.history["loss"]) - 1, linestyle="--", label="Fine-tuning")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_history.png", dpi=180)
    plt.close(fig)

    print(f"\nModel saved to: {model_path}")
    print(f"Evaluation files saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
