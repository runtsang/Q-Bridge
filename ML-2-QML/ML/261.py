"""Classical binary classifier with optional quantum fusion head.

The class exposes the same public API as the original QCNet but
allows a user‑supplied quantum module to be attached.  The
quantum part is optional and can be turned on via the ``use_quantum``
flag.  The module remains fully classical and is fully differentiable
with PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumHybridClassifier(nn.Module):
    """
    Classifier that operates purely classically or fuses a quantum
    expectation value as a secondary head.  The quantum head must
    be attached externally after construction via the ``quantum_module``
    attribute.
    """
    def __init__(self, in_features: int, use_quantum: bool = False, shift: float = 0.0):
        super().__init__()
        self.use_quantum = use_quantum
        self.shift = shift
        self.fc = nn.Linear(in_features, 1)
        # Placeholder for a quantum module that must provide a __call__
        # method returning a scalar tensor.
        self.quantum_module = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.fc(x)
        if self.use_quantum and self.quantum_module is not None:
            q_score = self.quantum_module(logits.squeeze(-1))
            logits = logits + self.shift * q_score
        probs = torch.sigmoid(logits)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["QuantumHybridClassifier"]
