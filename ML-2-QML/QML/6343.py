"""
Quantum quanvolution model with a trainable variational circuit per 2x2 patch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq

class QuanvolutionExtended(tq.QuantumModule):
    """
    Quantum quanvolution model with optional trainable variational circuit.
    """
    def __init__(self,
                 trainable: bool = True,
                 depth: int = 2,
                 device: str = "cpu"):
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
        if trainable:
            # Trainable parametric layer: repeated layers of rotations and CNOTs
            self.q_layer = tq.ParametricLayer(
                n_ops=8 * depth,
                wires=list(range(self.n_wires)),
                param_init="uniform",
            )
            self.q_layer.trainable = True
        else:
            self.q_layer = tq.RandomLayer(n_ops=8 * depth, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.linear = nn.Linear(4 * 14 * 14, 10)
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, 1, 28, 28)

        Returns:
            Log-softmax logits of shape (batch, 10)
        """
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=self.device)
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
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionExtended"]
