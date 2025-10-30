"""Quantum regression model using a quanvolution-inspired quantum kernel."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq


def generate_image_data(num_samples: int, img_size: int = 28) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic grayscale images and regression targets."""
    images = np.random.uniform(-1.0, 1.0, size=(num_samples, 1, img_size, img_size)).astype(np.float32)
    sums = images.sum(axis=(1, 2, 3))
    targets = np.sin(sums) + 0.1 * np.cos(2 * sums)
    return images, targets.astype(np.float32)


class RegressionDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples: int, img_size: int = 28):
        self.images, self.targets = generate_image_data(num_samples, img_size)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.images)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return {
            "image": torch.tensor(self.images[idx], dtype=torch.float32),
            "target": torch.tensor(self.targets[idx], dtype=torch.float32),
        }


class QuantumQuanvolutionFilter(tq.QuantumModule):
    """Quantum kernel applied to 2x2 patches of the image."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        # Encode each pixel into a qubit via Ry rotation
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


class HybridRegressionModel(tq.QuantumModule):
    """Quantum regression model: quantum quanvolution + linear head."""
    def __init__(self, img_size: int = 28):
        super().__init__()
        self.qfilter = QuantumQuanvolutionFilter()
        out_features = 4 * (img_size // 2) * (img_size // 2)
        self.head = nn.Linear(out_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        out = self.head(features)
        return out.squeeze(-1)

__all__ = ["HybridRegressionModel", "RegressionDataset", "generate_image_data"]
