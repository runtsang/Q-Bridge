"""Hybrid Quanvolution network with quantum layers.

The network mirrors the classical variant but replaces the convolutional
filter with a quantum 2‑qubit kernel and the fully‑connected layer with a
parameterised quantum circuit that outputs the expectation of Pauli‑Z.
Both components can be trained end‑to‑end on a quantum simulator.

The forward pass:
    1. Partition the image into 2×2 patches.
    2. Encode each pixel into a qubit via Ry(θ).
    3. Apply a random two‑qubit layer.
    4. Measure all qubits to obtain a 4‑dimensional feature vector.
    5. Compute a scalar quantum expectation from the feature vector
       using a single‑qubit Ry circuit.
    6. Concatenate the expectation with the feature vector and classify.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
from typing import Iterable

class QuantumFC(tq.QuantumModule):
    """Parameterised quantum circuit that returns the expectation of Pauli‑Z.

    The circuit consists of a single Ry rotation on one qubit, where the
    rotation angle is supplied by the input ``thetas``.  The expectation
    value is computed by measuring in the Z basis.
    """
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 1
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, thetas: torch.Tensor) -> torch.Tensor:
        bsz = thetas.shape[0]
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=thetas.device)
        # Apply Ry with the supplied angle for each batch element
        for i in range(bsz):
            qdev.ry(thetas[i], 0)
        measurement = self.measure(qdev)
        # expectation value of Z for each sample
        expectation = measurement.mean(dim=1)
        return expectation

class QuanvolutionFilter(tq.QuantumModule):
    """Quantum quanvolution filter that operates on 2×2 image patches.

    Each pixel of a patch is encoded into a qubit via Ry(θ), where θ is the
    pixel intensity.  A small random layer is applied before measuring all
    qubits.  The output is a 4‑dimensional feature vector per patch.
    """
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                data = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, data)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)

class HybridQuanvolutionNet(tq.QuantumModule):
    """Hybrid quantum network that combines a quanvolution filter,
    a quantum fully‑connected layer, and a classical classification head.

    The architecture parallels the classical variant but all feature
    extraction is performed on a quantum device.  The final linear layer
    is a standard PyTorch module, allowing easy integration with existing
    optimisers.
    """
    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.qfc = QuantumFC()
        # The linear head expects the 4×14×14 feature vector plus the
        # scalar expectation from the quantum FC.
        self.linear = nn.Linear(4 * 14 * 14 + 1, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)  # shape (bsz, 784)
        # Compute a scalar expectation from the mean of the features
        theta = torch.mean(features, dim=1, keepdim=True)
        qexpect = self.qfc(theta)  # shape (bsz,)
        qexpect = qexpect.unsqueeze(1)  # shape (bsz, 1)
        combined = torch.cat([features, qexpect], dim=1)
        logits = self.linear(combined)
        return F.log_softmax(logits, dim=-1)

__all__ = ["HybridQuanvolutionNet"]
