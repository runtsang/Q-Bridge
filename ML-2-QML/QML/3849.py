"""Hybrid CNN with quantum filter using TorchQuantum.

The quantum model mirrors the classical architecture:
  - Classical convolutional feature extractor (identical to the
    PyTorch version).
  - A 4‑qubit variational circuit that encodes a 2×2 image patch,
    applies a RandomLayer, and measures all qubits.  The output
    probability of measuring |1> per qubit is averaged and used
    as a scalar feature.
  - Concatenation of the quantum scalar with the flattened
    convolutional features.
  - Fully‑connected head producing four outputs, followed by
    BatchNorm1d for stability.

This construction keeps the same interface as the classical
model while providing a genuine quantum contribution.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class QuantumFilterLayer(tq.QuantumModule):
    """4‑qubit circuit that encodes a 2×2 image patch and outputs |1> probability."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        # Encoder maps 16 binary features into the 4 qubits.
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.random_layer = tq.RandomLayer(n_ops=20, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, 1, H, W) with pixel values in [0,1].
        Returns:
            Tensor of shape (B,) containing the average probability of
            measuring |1> across the 4 qubits for each sample.
        """
        bsz = x.size(0)
        # Extract a 2×2 patch from the top‑left corner
        patch = x[:, :, 0:2, 0:2]  # shape (B, 1, 2, 2)
        patch = patch.view(bsz, -1)  # shape (B, 4)
        # Threshold to binary and replicate to 16 bits
        binary = (patch > 0.5).float()
        binary = binary.repeat(1, 4)  # shape (B, 16)
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        self.encoder(qdev, binary)
        self.random_layer(qdev)
        out = self.measure(qdev)  # PauliZ expectation values
        # Convert PauliZ expectation to probability of |1>: (1 - Z)/2
        prob1 = (1 - out) / 2
        return prob1.mean(dim=1)  # shape (B,)


class ConvQFCModelQuantum(tq.QuantumModule):
    """Hybrid CNN + quantum filter + fully‑connected head using TorchQuantum."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.quantum_filter = QuantumFilterLayer()
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7 + 1, 64),  # +1 from quantum scalar
            nn.ReLU(inplace=True),
            nn.Linear(64, 4)
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.size(0)
        feats = self.features(x)           # shape (B, 16, 7, 7)
        flattened = feats.view(bsz, -1)    # shape (B, 16*7*7)
        q_scalar = self.quantum_filter(x)   # shape (B,)
        concat = torch.cat([flattened, q_scalar.unsqueeze(1)], dim=1)
        out = self.fc(concat)
        return self.norm(out)

__all__ = ["ConvQFCModelQuantum"]
