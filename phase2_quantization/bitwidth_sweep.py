"""
Phase 2 (bonus, for the report): sweep several Q(m, n) fixed-point formats
and report how test accuracy changes with each one. This gives you the data
needed for the report's "software model accuracy before/after quantization"
section, and shows WHY Q(8, 8) is a reasonable choice (not just an arbitrary
one).

Usage (with the venv activated, from inside phase2_quantization/):
    python bitwidth_sweep.py

Note: this runs the full fixed-point evaluation 5 times, so it can take
several minutes. That is expected -- it is intentionally pure-Python integer
arithmetic with no shortcuts, so it faithfully mirrors hardware behavior.
"""

import os
import sys

import torch

PHASE1_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phase1_software_training")
sys.path.insert(0, PHASE1_DIR)

import config                              # noqa: E402
from model import SimpleMLP                # noqa: E402
from data_utils import get_dataloaders     # noqa: E402

from fixed_point_utils import to_fixed, fixed_point_layer

# (QM, QN) pairs to try. All but the first two keep total width at 16 bits,
# so you can see how shifting the split between integer/fractional bits
# (not just total bit-width) affects accuracy.
FORMATS_TO_TEST = [
    (4, 4),    # 8-bit total   (very coarse, for comparison)
    (12, 4),   # 16-bit, integer-heavy
    (8, 8),    # 16-bit, balanced -- the project's suggested default
    (4, 12),   # 16-bit, fraction-heavy
    (16, 16),  # 32-bit total (near float32 precision, as an upper reference)
]


def quantize_matrix(matrix, qm, qn):
    return [[to_fixed(v, qm, qn) for v in row] for row in matrix]


def quantize_vector(vector, qm, qn):
    return [to_fixed(v, qm, qn) for v in vector]


def evaluate(model, test_loader, qm, qn):
    w_hidden_q = quantize_matrix(model.hidden.weight.detach().tolist(), qm, qn)
    b_hidden_q = quantize_vector(model.hidden.bias.detach().tolist(), qm, qn)
    w_output_q = quantize_matrix(model.output.weight.detach().tolist(), qm, qn)
    b_output_q = quantize_vector(model.output.bias.detach().tolist(), qm, qn)

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
    model = SimpleMLP()
    model.load_state_dict(torch.load(config.CHECKPOINT_PATH, map_location="cpu"))
    model.eval()

    _, test_loader = get_dataloaders()

    print(f"{'Format':<10}{'Total bits':<12}{'Accuracy':<10}")
    print("-" * 32)
    for qm, qn in FORMATS_TO_TEST:
        acc = evaluate(model, test_loader, qm, qn)
        print(f"Q({qm},{qn})".ljust(10) + f"{qm + qn}".ljust(12) + f"{acc * 100:.2f}%")


if __name__ == "__main__":
    main()
