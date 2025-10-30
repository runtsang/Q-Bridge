"""Quantum hybrid model with a trainable variational circuit.

The quantum branch encodes 2×2 image patches into a 4‑qubit system using a
GeneralEncoder, then passes the state through a parameterized random layer
(RandomLayer) with trainable parameters.  The measurement of all qubits yields a
feature vector that is concatenated across all patches and fed into a linear
classifier.  The trainable RandomLayer allows end‑to‑end learning of quantum
parameters while maintaining the same input/output shape as the original
QuanvolutionFilter.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq

class QuanvolutionHybrid(tq.QuantumModule):
    """Hybrid quantum‑classical model for MNIST using a trainable variational circuit."""

    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        # Encoder maps pixel values to Ry rotations
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        # Trainable random layer (variational circuit)
        self.q_layer = tq.RandomLayer(
            n_ops=8,
            wires=list(range(self.n_wires)),
            trainable=True
        )
        self.measure = tq.MeasureAll(tq.PauliZ)
        # Classical classifier
        self.classifier = nn.Linear(4 * 14 * 14, 10)

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
        features = torch.cat(patches, dim=1)
        logits = self.classifier(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionHybrid"]
