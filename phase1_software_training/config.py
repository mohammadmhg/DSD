"""
Central configuration file for the Neural Network Hardware Accelerator project.

This file defines constants that are shared across ALL phases of the project
(software training, quantization, VHDL hardware design, and testbench), so
that everyone -- Python code now, and VHDL code later -- works with the same
dimensions and file locations.
"""

import os

# ----------------------------------------------------------------------------
# Network architecture
# ----------------------------------------------------------------------------
# Original MNIST images are 28x28 = 784 pixels. That is too large for a
# simple, beginner-friendly hardware accelerator (784 MAC units per neuron
# would be huge). We downsample every image to IMG_SIZE x IMG_SIZE before
# feeding it to the network. This keeps the design small enough to implement
# and simulate in VHDL later, while still being a real, working classifier.
IMG_SIZE = 8                       # Downsampled image size (8x8 = 64 pixels)
INPUT_SIZE = IMG_SIZE * IMG_SIZE   # 64 inputs
HIDDEN_SIZE = 16                   # Neurons in the hidden layer
OUTPUT_SIZE = 10                   # 10 digit classes (0-9)

# ----------------------------------------------------------------------------
# Paths (all relative to this file, so the project can be moved anywhere)
# ----------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
PARAMS_DIR = os.path.join(OUTPUT_DIR, "params")
TEST_SAMPLES_DIR = os.path.join(OUTPUT_DIR, "test_samples")
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "model_checkpoint.pth")

# ----------------------------------------------------------------------------
# Training hyperparameters
# ----------------------------------------------------------------------------
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.01
RANDOM_SEED = 42

# Number of test samples to export for the future Testbench phase (Phase 4)
NUM_TEST_SAMPLES_TO_EXPORT = 10
