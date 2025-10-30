import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HybridBinaryClassifier(nn.Module):
    """Classical hybrid binary classifier.

    Combines a standard dense head with a simulated quantum‑like head
    (a sinusoidal non‑linearity). The two heads are fused by a learnable
    gating parameter that can shift the model behaviour from purely
    classical to purely quantum‑like during training.

    The class is fully differentiable and can be used in the same way
    as the original QCNet.
    """
    def __init__(self, in_features: int, gate_init: float = 0.5, shift: float = 0.0):
        super().__init__()
        self.classical_head = nn.Linear(in_features, 2)
        # quantum‑like head: a linear layer followed by a sine non‑linearity
        self.quantum_head = nn.Linear(in_features, 1)
        self.shift = shift
        self.gate = nn.Parameter(torch.tensor(gate_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # classical logits
        class_logits = self.classical_head(x)

        # quantum‑like logits
        q_logits = torch.sin(self.quantum_head(x) + self.shift)
        # expand to two logits by concatenating negative
        q_logits = torch.cat([q_logits, -q_logits], dim=-1)

        # fuse heads
        gate = torch.sigmoid(self.gate)
        logits = gate * class_logits + (1 - gate) * q_logits

        # convert to probabilities
        probs = F.softmax(logits, dim=-1)
        return probs

__all__ = ["HybridBinaryClassifier"]
