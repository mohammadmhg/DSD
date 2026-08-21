"""
Data loading and preprocessing utilities.

Responsible for:
  1. Downloading the MNIST dataset (handled automatically by torchvision,
     the first time you run it -- it caches the files under config.DATA_DIR).
  2. Downsampling images from 28x28 down to IMG_SIZE x IMG_SIZE.
  3. Producing PyTorch DataLoaders for training and testing.
"""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import config


def get_transform():
    """
    Build the preprocessing pipeline applied to every MNIST image:
      1. Resize from 28x28 down to config.IMG_SIZE x config.IMG_SIZE.
         This is the "image dimension reduction" step required by the
         project (mentioned in the final report structure).
      2. Convert to a PyTorch tensor with pixel values scaled to [0, 1].
    """
    return transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),  # scales pixels from [0, 255] to [0, 1]
    ])


def get_dataloaders():
    """
    Download (if needed) and load the MNIST train/test datasets, wrapped in
    DataLoaders ready for training and evaluation.

    Returns:
        train_loader, test_loader
    """
    transform = get_transform()

    train_dataset = datasets.MNIST(
        root=config.DATA_DIR, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=config.DATA_DIR, train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE, shuffle=False
    )

    return train_loader, test_loader
