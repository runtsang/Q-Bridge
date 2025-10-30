"""Hybrid quantum‑classical neural network for MNIST.

This module extends the original quanvolution idea by introducing
- a trainable variational ansatz that maps image patches to quantum
  feature vectors;
- a classical CNN backbone that processes the resulting feature map;
- a final linear classifier.

All components are differentiable using torchquantum and PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq

class QuanvolutionHybrid(tq.QuantumModule):
    """
    Quantum‑classical implementation of the quanvolution idea.
    The quantum part maps each 2×2 patch to a 4‑dim feature vector using
    a parameterized circuit.  The resulting feature map is then fed
    into a classical CNN backbone and a linear classifier.
    """
    def __init__(self, n_classes: int = 10) -> None:
        super().__init__()
        # Encode pixel intensities into qubit rotations
        self.encoder = tq.GeneralEncoder([
            {"input_idx": [0], "func": "ry", "wires": [0]},
            {"input_idx": [1], "func": "ry", "wires": [1]},
            {"input_idx": [2], "func": "ry", "wires": [2]},
            {"input_idx": [3], "func": "ry", "wires": [3]},
        ])
        # Trainable parameters of the variational circuit
        self.theta = nn.Parameter(torch.zeros(4))
        self.measure = tq.MeasureAll(tq.PauliZ)

        # Classical CNN backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        self.classifier = nn.Linear(576, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor of shape (batch, 1, 28, 28)

        Returns:
            log‑softmax over classes
        """
        batch_size = x.shape[0]
        x = x.view(batch_size, 28, 28)
        patches_list = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                # 4‑pixel patch
                data = torch.stack([
                    x[:, r, c],
                    x[:, r, c + 1],
                    x[:, r + 1, c],
                    x[:, r + 1, c + 1]
                ], dim=1)  # (batch, 4)
                qdev = tq.QuantumDevice(4, bsz=batch_size, device=x.device)
                # Encode pixel values
                self.encoder(qdev, data)
                # Apply parameterized variational circuit
                qdev.apply('ry', wires=[0], params=self.theta[0])
                qdev.apply('ry', wires=[1], params=self.theta[1])
                qdev.apply('ry', wires=[2], params=self.theta[2])
                qdev.apply('ry', wires=[3], params=self.theta[3])
                qdev.cx(0, 1)
                qdev.cx(1, 2)
                qdev.cx(2, 3)
                # Measurement
                measurement = self.measure(qdev)
                patches_list.append(measurement.view(batch_size, 4))
        # Concatenate all patch features
        features = torch.cat(patches_list, dim=1)  # (batch, 4*196)
        features = features.view(batch_size, 4, 14, 14)  # (batch, 4, 14, 14)
        # Classical backbone and classifier
        out = self.backbone(features)
        logits = self.classifier(out)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionHybrid"]
