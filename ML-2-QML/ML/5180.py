"""QuantumHybridNAT: purely classical hybrid architecture inspired by Quantum‑NAT.

The model fuses a CNN backbone with a classical variational block that mimics quantum
circuitry, and incorporates residual scaling/shift mechanisms from the fraud‑detection
example.  This allows end‑to‑end training with standard PyTorch optimizers while
preserving the expressive power of the original quantum designs.

The class is fully compatible with the original `QuantumNAT` API, accepting an
input tensor of shape `(batch, 1, 28, 28)` and returning logits for `num_classes`
classes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantumHybridNAT(nn.Module):
    """Classical hybrid network that mimics the Quantum‑NAT architecture.

    Components
    ----------
    * CNN backbone : 2 conv layers + pooling.
    * Projection to 4‑dim embedding (matches the number of quantum wires).
    * Residual scaling/shift (borrowed from FraudDetection).
    * Classical variational block that replaces the quantum circuit.
    * Final linear classifier.
    """

    def __init__(self, *, num_classes: int = 4, n_qubits: int = 4) -> None:
        super().__init__()
        # CNN backbone (same as the classical QFCModel)
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        # Project to 4‑dim vector (matches quantum wires)
        self.proj = nn.Linear(16 * 7 * 7, 4)

        # Residual scaling and shift (FraudDetection style)
        self.res_scale = nn.Parameter(torch.ones(4))
        self.res_shift = nn.Parameter(torch.zeros(4))

        # Classical variational block (mimics QLayer)
        self.vary = nn.Sequential(
            nn.Linear(4, 8),
            nn.Tanh(),
            nn.Linear(8, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
        )

        # Batch‑norm on the 4‑dim output
        self.batch_norm = nn.BatchNorm1d(4)

        # Final classifier
        self.out = nn.Linear(4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input image batch of shape (batch, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Log‑softmax logits of shape (batch, num_classes).
        """
        # Feature extraction
        feats = self.features(x)

        # 4‑dim embedding
        embed = self.proj(feats)

        # Residual connection
        residual = self.res_scale * embed + self.res_shift

        # Classical variational block
        var = self.vary(residual)

        # Combine and normalize
        combined = self.batch_norm(var)

        # Classifier
        logits = self.out(combined)

        return F.log_softmax(logits, dim=1)


__all__ = ["QuantumHybridNAT"]
