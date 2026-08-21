"""
MLP model definition.

Architecture (kept intentionally small so it can later be built in hardware):

    Input (64) -> Linear -> ReLU -> Hidden (16) -> Linear -> Output (10)

Note: we do NOT apply Softmax at the end. During training we use
CrossEntropyLoss, which internally applies Softmax for us. During inference
and export we simply take the argmax of the raw output ("logits") -- this is
exactly what the hardware will do too later (compare 10 output values and
pick the index of the largest one).
"""

import torch.nn as nn

import config


class SimpleMLP(nn.Module):
    """A minimal 1-hidden-layer MLP for downsampled MNIST digit classification."""

    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(config.INPUT_SIZE, config.HIDDEN_SIZE)
        self.relu = nn.ReLU()
        self.output = nn.Linear(config.HIDDEN_SIZE, config.OUTPUT_SIZE)

    def forward(self, x):
        # x arrives as (batch, 1, IMG_SIZE, IMG_SIZE); flatten to (batch, INPUT_SIZE)
        x = x.view(x.size(0), -1)
        x = self.relu(self.hidden(x))
        x = self.output(x)  # raw logits, no softmax (matches hardware behavior)
        return x
