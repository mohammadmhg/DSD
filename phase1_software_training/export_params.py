"""
Export trained model parameters (weights & biases) to plain text files.

These files will be used in Phase 2 (quantization to fixed-point) and
Phase 3 (loading values into VHDL ROMs / registers), so the format is kept
simple on purpose: one decimal number per line, in row-major order.

Usage (AFTER train.py has produced a checkpoint):
    python export_params.py
"""

import os

import torch

import config
from model import SimpleMLP


def save_tensor_as_txt(tensor, filepath):
    """
    Save a PyTorch tensor to a plain text file, one value per line.

    2D tensors (weight matrices) are flattened in row-major order, so for
    the hidden layer, row 0 = all 64 input weights of hidden neuron 0,
    row 1 = all 64 input weights of hidden neuron 1, and so on.
    """
    flat = tensor.detach().cpu().numpy().flatten()
    with open(filepath, "w") as f:
        for value in flat:
            f.write(f"{value:.8f}\n")
    print(f"Saved {tuple(tensor.shape)} -> {filepath} ({flat.size} values)")


def export():
    os.makedirs(config.PARAMS_DIR, exist_ok=True)

    model = SimpleMLP()
    model.load_state_dict(torch.load(config.CHECKPOINT_PATH, map_location="cpu"))
    model.eval()

    # Hidden layer: weight shape (HIDDEN_SIZE, INPUT_SIZE), bias shape (HIDDEN_SIZE,)
    save_tensor_as_txt(
        model.hidden.weight, os.path.join(config.PARAMS_DIR, "weights_hidden.txt")
    )
    save_tensor_as_txt(
        model.hidden.bias, os.path.join(config.PARAMS_DIR, "bias_hidden.txt")
    )

    # Output layer: weight shape (OUTPUT_SIZE, HIDDEN_SIZE), bias shape (OUTPUT_SIZE,)
    save_tensor_as_txt(
        model.output.weight, os.path.join(config.PARAMS_DIR, "weights_output.txt")
    )
    save_tensor_as_txt(
        model.output.bias, os.path.join(config.PARAMS_DIR, "bias_output.txt")
    )

    # Save a small metadata file describing shapes/order -- needed as a
    # reference in later phases (quantization and VHDL ROM generation).
    metadata_path = os.path.join(config.PARAMS_DIR, "README_params.txt")
    with open(metadata_path, "w") as f:
        f.write("Parameter export format\n")
        f.write("========================\n")
        f.write(f"INPUT_SIZE  = {config.INPUT_SIZE}\n")
        f.write(f"HIDDEN_SIZE = {config.HIDDEN_SIZE}\n")
        f.write(f"OUTPUT_SIZE = {config.OUTPUT_SIZE}\n\n")
        f.write("weights_hidden.txt : HIDDEN_SIZE x INPUT_SIZE values, row-major\n")
        f.write("                     (row h = weights of hidden neuron h)\n")
        f.write("bias_hidden.txt    : HIDDEN_SIZE values\n")
        f.write("weights_output.txt : OUTPUT_SIZE x HIDDEN_SIZE values, row-major\n")
        f.write("bias_output.txt    : OUTPUT_SIZE values\n")
        f.write("All values are plain decimal floating point (one per line).\n")
        f.write("Phase 2 will convert these into fixed-point Q(m,n) format.\n")
    print(f"Saved parameter format description -> {metadata_path}")


if __name__ == "__main__":
    export()
