"""
Export a handful of test images, their true labels, and the model's
predicted labels, for later use in Phase 4 (VHDL Testbench).

Usage (AFTER train.py has produced a checkpoint):
    python export_test_samples.py
"""

import os

import torch

import config
from data_utils import get_dataloaders
from model import SimpleMLP


def export():
    os.makedirs(config.TEST_SAMPLES_DIR, exist_ok=True)

    _, test_loader = get_dataloaders()
    images, labels = next(iter(test_loader))
    images = images[: config.NUM_TEST_SAMPLES_TO_EXPORT]
    labels = labels[: config.NUM_TEST_SAMPLES_TO_EXPORT]

    model = SimpleMLP()
    model.load_state_dict(torch.load(config.CHECKPOINT_PATH, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        predictions = model(images).argmax(dim=1)

    images_path = os.path.join(config.TEST_SAMPLES_DIR, "test_images.txt")
    labels_path = os.path.join(config.TEST_SAMPLES_DIR, "test_labels.txt")
    predictions_path = os.path.join(config.TEST_SAMPLES_DIR, "python_predictions.txt")

    # Each line = one image: INPUT_SIZE pixel values separated by spaces
    with open(images_path, "w") as f:
        for img in images:
            pixels = img.view(-1).numpy()
            f.write(" ".join(f"{p:.8f}" for p in pixels) + "\n")

    with open(labels_path, "w") as f:
        for label in labels:
            f.write(f"{label.item()}\n")

    with open(predictions_path, "w") as f:
        for pred in predictions:
            f.write(f"{pred.item()}\n")

    print(f"Saved {len(images)} test images  -> {images_path}")
    print(f"Saved {len(images)} true labels  -> {labels_path}")
    print(f"Saved {len(images)} predictions  -> {predictions_path}")
    print("\nThese three files will become the Testbench inputs/expected outputs in Phase 4.")


if __name__ == "__main__":
    export()
