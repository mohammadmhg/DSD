"""
Phase 2: Quantization to fixed-point Q(m, n) format, and accuracy evaluation.

What this script does:
  1. Loads the Phase 1 trained model checkpoint.
  2. Quantizes all weights and biases to Q(QM, QN) fixed-point integers.
  3. Runs the ENTIRE forward pass using ONLY fixed-point integer arithmetic
     (see fixed_point_utils.fixed_point_layer) -- exactly mimicking what the
     VHDL hardware will do in Phase 3 -- and measures accuracy on the MNIST
     test set.
  4. Compares this fixed-point accuracy against the original float accuracy,
     so you can report the accuracy drop caused by quantization.
  5. Exports the quantized weights/biases (decimal + hex) to
     outputs/params_fixed/  -> ready to become VHDL ROM contents in Phase 3.
  6. Exports a few quantized test samples + fixed-point predictions to
     outputs/test_samples_fixed/ -> ready for the Phase 4 Testbench.

Usage (with the venv activated, from inside phase2_quantization/):
    python quantize_and_evaluate.py

Note: evaluating on the full 10,000-image MNIST test set using pure-Python
fixed-point arithmetic (on purpose -- no NumPy/PyTorch shortcuts, so it truly
mirrors integer hardware) takes roughly 1-2 minutes. This is expected.
"""

import os
import sys

import torch

# ----------------------------------------------------------------------------
# Make Phase 1's modules (config, model, data_utils) importable from here,
# without duplicating any code or files.
# ----------------------------------------------------------------------------
PHASE1_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phase1_software_training")
sys.path.insert(0, PHASE1_DIR)

import config                              # noqa: E402  (from phase1_software_training)
from model import SimpleMLP                # noqa: E402
from data_utils import get_dataloaders     # noqa: E402

from fixed_point_utils import to_fixed, to_hex, fixed_point_layer

# ----------------------------------------------------------------------------
# Quantization format: Q(QM, QN).
# QM = integer bits (including sign), QN = fractional bits.
# Default matches the example given in the project spec: Q(8, 8), 16-bit total.
# ----------------------------------------------------------------------------
QM = 8
QN = 8

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(THIS_DIR, "outputs")
PARAMS_FIXED_DIR = os.path.join(OUTPUT_DIR, "params_fixed")
TEST_SAMPLES_FIXED_DIR = os.path.join(OUTPUT_DIR, "test_samples_fixed")


def quantize_matrix(matrix, qm, qn):
    """Quantize a 2D list of floats to a 2D list of fixed-point integers."""
    return [[to_fixed(v, qm, qn) for v in row] for row in matrix]


def quantize_vector(vector, qm, qn):
    """Quantize a 1D list of floats to a list of fixed-point integers."""
    return [to_fixed(v, qm, qn) for v in vector]


def save_fixed_params(name, values, qm, qn, out_dir):
    """
    Save a quantized parameter tensor to two text files:
      <name>_dec.txt : one signed decimal integer per line
      <name>_hex.txt : one two's-complement hex string per line
    Flattened in the same row-major order used in Phase 1's export_params.py,
    so the two phases stay consistent.
    """
    flat = []
    for row in values:
        if isinstance(row, list):
            flat.extend(row)
        else:
            flat.append(row)

    dec_path = os.path.join(out_dir, f"{name}_dec.txt")
    hex_path = os.path.join(out_dir, f"{name}_hex.txt")
    with open(dec_path, "w") as fd, open(hex_path, "w") as fh:
        for v in flat:
            fd.write(f"{v}\n")
            fh.write(f"{to_hex(v, qm, qn)}\n")
    print(f"Saved {len(flat)} values -> {dec_path}  and  {hex_path}")


def load_trained_model():
    model = SimpleMLP()
    model.load_state_dict(torch.load(config.CHECKPOINT_PATH, map_location="cpu"))
    model.eval()
    return model


def evaluate_float_accuracy(model, test_loader):
    """Baseline accuracy using the original float32 model (same measurement as Phase 1)."""
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


def evaluate_fixed_point_accuracy(test_loader, w_hidden_q, b_hidden_q, w_output_q, b_output_q, qm, qn):
    """
    Accuracy using ONLY fixed-point integer arithmetic for the forward pass.
    This is the number that predicts what the real VHDL hardware will achieve.
    """
    correct, total = 0, 0
    for images, labels in test_loader:
        for img, label in zip(images, labels):
            pixels_q = [to_fixed(p, qm, qn) for p in img.view(-1).tolist()]

            hidden_q = fixed_point_layer(pixels_q, w_hidden_q, b_hidden_q, qm, qn, apply_relu=True)
            output_q = fixed_point_layer(hidden_q, w_output_q, b_output_q, qm, qn, apply_relu=False)

            predicted = output_q.index(max(output_q))
            correct += int(predicted == label.item())
            total += 1
    return correct / total


def main():
    os.makedirs(PARAMS_FIXED_DIR, exist_ok=True)
    os.makedirs(TEST_SAMPLES_FIXED_DIR, exist_ok=True)

    print(f"Quantization format: Q({QM}, {QN})  ->  {QM + QN}-bit signed fixed-point\n")

    model = load_trained_model()

    # ---- 1. Quantize weights & biases, and save them (decimal + hex) ----
    w_hidden_q = quantize_matrix(model.hidden.weight.detach().tolist(), QM, QN)
    b_hidden_q = quantize_vector(model.hidden.bias.detach().tolist(), QM, QN)
    w_output_q = quantize_matrix(model.output.weight.detach().tolist(), QM, QN)
    b_output_q = quantize_vector(model.output.bias.detach().tolist(), QM, QN)

    save_fixed_params("weights_hidden", w_hidden_q, QM, QN, PARAMS_FIXED_DIR)
    save_fixed_params("bias_hidden", b_hidden_q, QM, QN, PARAMS_FIXED_DIR)
    save_fixed_params("weights_output", w_output_q, QM, QN, PARAMS_FIXED_DIR)
    save_fixed_params("bias_output", b_output_q, QM, QN, PARAMS_FIXED_DIR)

    # ---- 2. Evaluate accuracy: float baseline vs fixed-point ----
    _, test_loader = get_dataloaders()

    float_acc = evaluate_float_accuracy(model, test_loader)
    print("\nEvaluating fixed-point accuracy (pure integer arithmetic, ~1-2 min)...")
    fixed_acc = evaluate_fixed_point_accuracy(test_loader, w_hidden_q, b_hidden_q, w_output_q, b_output_q, QM, QN)

    print("\n===== Accuracy comparison =====")
    print(f"Float32 accuracy                  : {float_acc * 100:.2f}%")
    print(f"Fixed-point Q({QM},{QN}) accuracy         : {fixed_acc * 100:.2f}%")
    print(f"Accuracy drop from quantization    : {(float_acc - fixed_acc) * 100:.2f} percentage points")

    # ---- 3. Export a few quantized test samples for the Phase 4 Testbench ----
    images, labels = next(iter(test_loader))
    images = images[: config.NUM_TEST_SAMPLES_TO_EXPORT]
    labels = labels[: config.NUM_TEST_SAMPLES_TO_EXPORT]

    images_path = os.path.join(TEST_SAMPLES_FIXED_DIR, "test_images_fixed_dec.txt")
    predictions_path = os.path.join(TEST_SAMPLES_FIXED_DIR, "hardware_predictions.txt")
    labels_path = os.path.join(TEST_SAMPLES_FIXED_DIR, "test_labels.txt")

    with open(images_path, "w") as f_img, open(predictions_path, "w") as f_pred, open(labels_path, "w") as f_lab:
        for img, label in zip(images, labels):
            pixels_q = [to_fixed(p, QM, QN) for p in img.view(-1).tolist()]
            f_img.write(" ".join(str(v) for v in pixels_q) + "\n")

            hidden_q = fixed_point_layer(pixels_q, w_hidden_q, b_hidden_q, QM, QN, apply_relu=True)
            output_q = fixed_point_layer(hidden_q, w_output_q, b_output_q, QM, QN, apply_relu=False)
            predicted = output_q.index(max(output_q))

            f_pred.write(f"{predicted}\n")
            f_lab.write(f"{label.item()}\n")

    print(f"\nSaved {len(images)} quantized test images + predictions -> {TEST_SAMPLES_FIXED_DIR}")
    print("These integers are exactly what Phase 3's VHDL simulation should reproduce.")

    # ---- 4. Save a metadata file describing the fixed-point format ----
    meta_path = os.path.join(PARAMS_FIXED_DIR, "README_fixed_params.txt")
    with open(meta_path, "w") as f:
        f.write("Fixed-point parameter export format\n")
        f.write("====================================\n")
        f.write(f"Format       : Q({QM}, {QN})  -- {QM} integer bits (incl. sign) + {QN} fractional bits\n")
        f.write(f"Total width  : {QM + QN} bits, two's complement\n")
        f.write(f"Scale factor : 2^{QN} = {1 << QN}\n\n")
        f.write("*_dec.txt : one signed decimal integer per line\n")
        f.write("*_hex.txt : one two's-complement hex string per line (paste into a VHDL ROM)\n\n")
        f.write("weights_hidden_* : HIDDEN_SIZE x INPUT_SIZE values, row-major\n")
        f.write("bias_hidden_*    : HIDDEN_SIZE values\n")
        f.write("weights_output_* : OUTPUT_SIZE x HIDDEN_SIZE values, row-major\n")
        f.write("bias_output_*    : OUTPUT_SIZE values\n")
    print(f"Saved format description -> {meta_path}")


if __name__ == "__main__":
    main()
