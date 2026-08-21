"""
Training script for the MLP digit classifier.

Usage (from inside phase1_software_training/, with the virtual environment
activated):
    python train.py

What it does:
  1. Loads and preprocesses MNIST (downloads it automatically the first time).
  2. Trains the SimpleMLP model for config.EPOCHS epochs.
  3. Evaluates accuracy on the test set after every epoch.
  4. Saves the trained model weights to config.CHECKPOINT_PATH.
"""

import os

import torch
import torch.nn as nn
import torch.optim as optim

import config
from data_utils import get_dataloaders
from model import SimpleMLP


def evaluate(model, test_loader, device):
    """Compute classification accuracy on the test set."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    return correct / total


def train():
    torch.manual_seed(config.RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    train_loader, test_loader = get_dataloaders()

    model = SimpleMLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.LEARNING_RATE, momentum=0.9)

    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        avg_loss = running_loss / len(train_loader.dataset)
        test_acc = evaluate(model, test_loader, device)
        print(
            f"Epoch {epoch:2d}/{config.EPOCHS} | "
            f"Loss: {avg_loss:.4f} | Test Accuracy: {test_acc * 100:.2f}%"
        )

    torch.save(model.state_dict(), config.CHECKPOINT_PATH)
    print(f"\nModel checkpoint saved to: {config.CHECKPOINT_PATH}")


if __name__ == "__main__":
    train()
