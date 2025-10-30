"""Quantum quanvolution with a trainable variational circuit for each 2×2 patch."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq

class QuanvolutionFilter(tq.QuantumModule):
    """Quantum filter that applies a variational circuit to each 2×2 patch."""

    def __init__(self, n_wires: int = 4, n_layers: int = 3):
        super().__init__()
        self.n_wires = n_wires
        self.n_layers = n_layers
        # Parameterized quantum circuit
        self.circuit = tq.ParameterizedQuantumCircuit(n_wires, n_layers, use_cnot=True)
        # Measurement in Z basis
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        # Reshape to (batch, 28, 28)
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                # Extract 2x2 patch and flatten
                patch = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                # Encode patch into qubits
                self.circuit(qdev, patch)
                # Measure
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        # Concatenate all patch measurements
        return torch.cat(patches, dim=1)

class QuanvolutionClassifier(nn.Module):
    """Hybrid model using the quantum filter followed by a linear head."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.linear = nn.Linear(4 * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier"]
