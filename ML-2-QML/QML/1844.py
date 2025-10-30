"""Quantum implementation of a hybrid quanvolution architecture.

This module defines QuanvolutionHybrid, which processes 2×2 patches with a
trainable 4‑qubit variational circuit and then refines the resulting
feature map with a classical residual block before classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq

class QuantumPatchEncoder(tq.QuantumModule):
    """Trainable quantum encoder for a 2×2 image patch."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        # Encode each pixel into a rotation around Y
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        # Trainable variational layer
        self.var_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, qdev: tq.QuantumDevice, data: torch.Tensor) -> torch.Tensor:
        # Encode classical data
        self.encoder(qdev, data)
        # Apply trainable variational circuit
        self.var_layer(qdev)
        # Measure all qubits
        return self.measure(qdev)

class QuantumPatchLayer(tq.QuantumModule):
    """Applies a quantum filter over all 2×2 patches of the input image."""
    def __init__(self):
        super().__init__()
        self.patch_encoder = QuantumPatchEncoder()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, _, h, w = x.shape
        device = x.device
        qdev = tq.QuantumDevice(self.patch_encoder.n_wires, bsz=bsz, device=device)
        patches = []
        for r in range(0, h, 2):
            for c in range(0, w, 2):
                patch = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                measurement = self.patch_encoder(qdev, patch)
                patches.append(measurement.view(bsz, 4))
        # Concatenate all patches into a flat feature vector per sample
        return torch.cat(patches, dim=1)

class ResidualBlock(nn.Module):
    """Classical residual block applied after the quantum filter."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if in_channels!= out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out

class QuanvolutionHybrid(nn.Module):
    """Quantum‑classical hybrid model that replaces the 2×2 conv filter with a
    trainable quantum circuit.
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.qfilter = QuantumPatchLayer()
        # The output of qfilter is a flat vector of length 4*14*14
        # Reshape to (B, 4, 14, 14) before the residual block
        self.res_block = ResidualBlock(4, 4)
        self.classifier = nn.Linear(4 * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is (B, 1, 28, 28)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        # Quantum filter produces a flat feature vector
        features = self.qfilter(x)  # (B, 4*14*14)
        # Reshape for residual block
        features = features.view(x.size(0), 4, 14, 14)
        features = self.res_block(features)
        # Flatten again before classification
        features = features.view(x.size(0), -1)
        logits = self.classifier(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionHybrid"]
