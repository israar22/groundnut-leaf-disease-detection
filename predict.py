"""Predict the class of a groundnut leaf image."""
import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf

from src.model_utils import prepare_image, load_class_names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Path to a leaf image")
    parser.add_argument("--model", default="models/groundnut_efficientnetv2b0.keras")
    parser.add_argument("--classes", default="models/class_names.json")
    args = parser.parse_args()

    if not Path(args.model).is_file():
        raise FileNotFoundError(f"Model not found: {args.model}. Train the model first.")
    if not Path(args.classes).is_file():
        raise FileNotFoundError(f"Class mapping not found: {args.classes}. Train the model first.")

    model = tf.keras.models.load_model(args.model)
    class_names = load_class_names(args.classes)
    image_batch = prepare_image(args.image)
    scores = model.predict(image_batch, verbose=0)[0]
    best = int(np.argmax(scores))

    print(f"Prediction: {class_names[best]}")
    print(f"Softmax score: {scores[best] * 100:.2f}%")
    print("\nAll class scores:")
    for name, score in sorted(zip(class_names, scores), key=lambda item: item[1], reverse=True):
        print(f"  {name}: {score * 100:.2f}%")


if __name__ == "__main__":
    main()
