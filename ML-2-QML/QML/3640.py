"""Quantum regression model that encodes 2‑D image features and applies a variational circuit."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torch.utils.data import Dataset

def generate_image_data(num_samples: int, img_size: int = 28, noise_level: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """
    Same image generation routine as the classical counterpart.
    The function is duplicated to keep the API identical across modules.
    """
    x = np.random.uniform(-1.0, 1.0, size=(num_samples, 2)).astype(np.float32)
    angles = 3 * x[:, 0] + 2 * x[:, 1]
    images = np.sin(angles[:, None, None] + np.arange(img_size)[:, None] * 0.1
                    + np.arange(img_size)[None, :] * 0.1)
    images = images.reshape(num_samples, 1, img_size, img_size).astype(np.float32)
    images += np.random.normal(scale=noise_level, size=images.shape).astype(np.float32)
    images = np.clip(images, -1.0, 1.0)
    targets = np.sin(2 * x[:, 0]) * np.cos(x[:, 1])
    return images, targets.astype(np.float32)

class RegressionDataset(Dataset):
    """
    Dataset for the quantum model.
    Wraps the same image generation function but returns tensors
    suitable for amplitude‑encoding into a quantum device.
    """
    def __init__(self, samples: int, img_size: int = 28):
        self.images, self.targets = generate_image_data(samples, img_size)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.images)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "image": torch.tensor(self.images[index], dtype=torch.float32),
            "target": torch.tensor(self.targets[index], dtype=torch.float32),
        }

class QModel(tq.QuantumModule):
    """
    Variational quantum regression model.

    Encoder uses a 4×4_ryzxy pattern to embed the flattened image into
    a 16‑wire quantum state.  A random layer with 50 operations
    followed by trainable single‑qubit gates expands the expressivity.
    The output of a Pauli‑Z measurement is batch‑normalised before a
    classical linear head produces the regression score.
    """
    class QLayer(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.n_wires = 16
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            for w in range(self.n_wires):
                self.rx(qdev, wires=w)
                self.ry(qdev, wires=w)
                self.rz(qdev, wires=w)

    def __init__(self):
        super().__init__()
        self.n_wires = 16
        # Encoder that maps a 16‑dimensional feature vector to the quantum state
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)
        self.head = nn.Linear(self.n_wires, 1)

    def forward(self, image_batch: torch.Tensor) -> torch.Tensor:
        # Flatten the image to a 16‑dimensional vector via simple pooling
        pooled = torch.nn.functional.avg_pool2d(image_batch, kernel_size=8).view(image_batch.shape[0], -1)
        bsz = pooled.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=image_batch.device, record_op=True)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        features = self.measure(qdev)
        out = self.norm(features)
        return self.head(out).squeeze(-1)

__all__ = ["QModel", "RegressionDataset", "generate_image_data"]
