"""Unified Quanvolution quantum module.

This module builds on the original QuanvolutionFilter but adds a
parameterised random circuit and a measurement that outputs a 4‑dim
feature vector per 2×2 patch.  The design is compatible with
torchquantum and can be swapped with any other variational circuit.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq

class UnifiedQuanvolutionFilter(tq.QuantumModule):
    """Quantum filter that processes 2×2 image patches via a small
    parameterised circuit.  The circuit consists of an Ry encoder
    followed by a random 2‑layer layer and a measurement of all qubits."""
    def __init__(
        self,
        kernel_size: int = 2,
        n_ops: int = 8,
        n_wires: int | None = None,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.n_wires = n_wires or kernel_size ** 2

        # Encoder: map each pixel to an Ry gate
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "ry", "wires": [i]}
                for i in range(self.n_wires)
            ]
        )

        # Random variational layer
        self.q_layer = tq.RandomLayer(n_ops=n_ops, wires=list(range(self.n_wires)))

        # Measurement of all qubits in Z basis
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (N, 1, 28, 28)

        Returns:
            Tensor of shape (N, 4 * 14 * 14) containing the
            measurement results for each patch.
        """
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)

        # Reshape to patches
        patches = x.unfold(2, self.kernel_size, self.kernel_size) \
                     .unfold(3, self.kernel_size, self.kernel_size)  # (N, 1, 14, 14, 2, 2)
        patches = patches.contiguous().view(-1, self.n_wires)  # (N*196, 4)

        # Process each patch
        outputs = []
        for data in patches:
            self.encoder(qdev, data)
            self.q_layer(qdev)
            measurement = self.measure(qdev)  # (N*196, 4)
            outputs.append(measurement)

        measurements = torch.cat(outputs, dim=0)  # (N*196, 4)
        return measurements.view(bsz, -1)

class UnifiedQuanvolutionClassifier(nn.Module):
    """Hybrid classifier that uses the quantum filter followed by a linear head."""
    def __init__(self) -> None:
        super().__init__()
        self.qfilter = UnifiedQuanvolutionFilter()
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["UnifiedQuanvolutionFilter", "UnifiedQuanvolutionClassifier"]
