"""Hybrid quantum regression model.

The quantum part implements a quanvolution filter that processes
2×2 patches of a 28×28 image.  The resulting 4‑qubit feature vectors
are fed into a small, trainable quantum circuit that outputs a
single expectation value per patch.  The patch‑level predictions are
averaged and passed through a linear head to obtain the final
continuous target."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq

def generate_image_regression_data(samples: int, image_size: int = 28) -> tuple[np.ndarray, np.ndarray]:
    images = np.random.rand(samples, 1, image_size, image_size).astype(np.float32)
    targets = images.sum(axis=(1, 2, 3)) + 0.1 * np.sin(images.sum(axis=(1, 2, 3)))
    return images, targets.astype(np.float32)

class HybridRegressionDataset(torch.utils.data.Dataset):
    def __init__(self, samples: int, image_size: int = 28):
        self.images, self.targets = generate_image_regression_data(samples, image_size)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.images)

    def __getitem__(self, idx: int):
        return {"image": torch.tensor(self.images[idx], dtype=torch.cfloat),
                "target": torch.tensor(self.targets[idx], dtype=torch.float32)}

class QuanvolutionFilter(tq.QuantumModule):
    """Random 4‑qubit kernel applied to each 2×2 patch of a 28×28 image."""
    def __init__(self):
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
        self.q_layer = tq.RandomLayer(n_ops=12, wires=list(range(self.n_wires)))
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

class RegressionQuantumLayer(tq.QuantumModule):
    """Small parameterised quantum circuit that produces a single expectation value."""
    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{n_wires}xRy"])
        self.q_layer = tq.RandomLayer(n_ops=10, wires=list(range(n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
        self.encoder(qdev)
        self.q_layer(qdev)
        # Return mean of Z‑measurements for each qubit
        return self.measure(qdev).mean(dim=-1, keepdim=True)

class HybridQuantumRegression(tq.QuantumModule):
    def __init__(self, image_size: int = 28):
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.qreg_layer = RegressionQuantumLayer(4)
        self.head = nn.Linear(1, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        patch_features = self.qfilter(image)  # shape (bsz, 4*14*14)
        bsz = patch_features.shape[0]
        n_patches = patch_features.shape[1] // 4
        patch_features = patch_features.view(bsz, n_patches, 4)
        # Flatten patches for quantum device
        qdev = tq.QuantumDevice(4, bsz=n_patches * bsz, device=image.device)
        qdev.set_state(patch_features.reshape(-1, 4))
        patch_preds = self.qreg_layer(qdev)  # shape (n_patches*bsz, 1)
        patch_preds = patch_preds.view(bsz, n_patches)
        mean_pred = patch_preds.mean(dim=1, keepdim=True)  # shape (bsz, 1)
        return self.head(mean_pred).squeeze(-1)

__all__ = ["HybridRegressionDataset", "HybridQuantumRegression"]
