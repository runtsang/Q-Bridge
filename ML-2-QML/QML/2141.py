"""Quantum quanvolution module inspired by the original Quanvolution example."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq

class QuanvolutionHybrid(tq.QuantumModule):
    """
    Quantum quanvolution filter and classifier.
    Uses a 2×2 image patch encoder into 4 qubits, followed by a trainable variational circuit.
    The measurement of all qubits forms the feature vector, which is passed to a classical linear head.
    Supports temperature scaling and optional ensemble of multiple circuits for uncertainty.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_classes: int = 10,
        n_wires: int = 4,
        n_params: int = 8,
        ensemble_size: int = 1,
        dropout_prob: float = 0.0,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.ensemble_size = ensemble_size
        self.dropout = nn.Dropout(dropout_prob)
        self.measure = tq.MeasureAll(tq.PauliZ)

        # Encoder maps pixel intensities to rotation angles
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

        # Variational layer with trainable parameters
        self.var_layer = tq.RandomLayer(n_ops=n_params, wires=list(range(self.n_wires)))
        self.temperature = nn.Parameter(torch.tensor(temperature))

        # Classical head
        feature_dim = n_wires * 14 * 14 * ensemble_size
        self.fc = nn.Linear(feature_dim, out_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the quantum filter and linear head.
        x: (batch, 1, 28, 28)
        returns log probabilities (batch, out_classes)
        """
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)

        # Extract 2×2 patches
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, patch)
                self.var_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, self.n_wires))

        features = torch.cat(patches, dim=1)
        # If ensemble_size > 1, replicate features
        if self.ensemble_size > 1:
            features = features.repeat(1, self.ensemble_size)

        features = self.dropout(features)
        logits = self.fc(features)
        logits = logits / self.temperature
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionHybrid"]
