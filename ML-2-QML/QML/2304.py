"""Quantum hybrid regression module with a quanvolution front‑end.

The quantum version replaces the classical patch extractor with a
quantum kernel that processes each 2×2 patch.  A variational circuit
then learns a representation of the averaged patch features, and a
linear head produces the regression output.
"""

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torch.utils.data import Dataset

def generate_superposition_patch_data(
    num_samples: int,
    patch_size: int = 2,
    image_size: int = 28,
    channel: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic images and a regression target.

    Images are filled with random values in ``[-1, 1]``.  The target for
    each image is computed as ``sin(sum) + 0.1*cos(2*sum)`` to mimic
    the superposition‑like function used in the classical seed.
    """
    images = np.random.uniform(-1.0, 1.0, size=(num_samples, channel, image_size, image_size)).astype(np.float32)
    sums = images.reshape(num_samples, -1).sum(axis=1)
    labels = np.sin(sums) + 0.1 * np.cos(2 * sums)
    return images, labels.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset returning images and regression targets for the quantum model."""
    def __init__(self, samples: int, patch_size: int = 2, image_size: int = 28, channel: int = 1):
        self.images, self.labels = generate_superposition_patch_data(samples, patch_size, image_size, channel)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.images)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.images[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class QuanvolutionFilter(tq.QuantumModule):
    """Quantum kernel applied to each 2×2 patch.

    The filter uses a random two‑qubit circuit followed by a measurement
    in the Pauli‑Z basis.  The result is a 4‑dimensional feature vector
    per patch, matching the classical patch size.
    """
    def __init__(self, n_wires: int = 4, n_ops: int = 8):
        super().__init__()
        self.n_wires = n_wires
        # Encode each pixel with an Ry gate
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.random_layer = tq.RandomLayer(n_ops=n_ops, wires=list(range(n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Batch of patches of shape ``(B, 4)`` where each row contains
            the four pixel values of a 2×2 patch.

        Returns
        -------
        torch.Tensor
            Quantum‑kernel features of shape ``(B, 4)``.
        """
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)

        self.encoder(qdev, x)
        self.random_layer(qdev)
        return self.measure(qdev).view(bsz, self.n_wires)

class HybridRegressionModel(tq.QuantumModule):
    """Quantum hybrid regression model with a quanvolution front‑end.

    The architecture mirrors the classical version but replaces the
    patch extractor with a quantum kernel.  The flattened patch
    features are then processed by a variational circuit before a
    linear head outputs the regression target.
    """
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(n_wires)))
            # Two trainable single‑qubit rotations per wire
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            for w in range(self.n_wires):
                self.rx(qdev, wires=w)
                self.ry(qdev, wires=w)

    def __init__(self, patch_size: int = 2, hidden_dim: int = 64, n_wires: int = 4):
        super().__init__()
        self.patch_size = patch_size
        self.n_wires = n_wires
        self.qfilter = QuanvolutionFilter(n_wires=n_wires)
        self.q_layer = self.QLayer(n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(n_wires, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Batch of images of shape ``(B, 1, 28, 28)``.

        Returns
        -------
        torch.Tensor
            Regression output of shape ``(B,)``.
        """
        bsz, _, h, w = x.shape
        # Extract 2×2 patches
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(bsz, -1, self.patch_size * self.patch_size)  # (B, P, 4)

        # Apply quantum kernel to each patch
        patch_features = self.qfilter(patches.view(-1, self.patch_size * self.patch_size))
        patch_features = patch_features.view(bsz, -1, self.n_wires)  # (B, P, 4)

        # Average across patches to obtain a single 4‑dimensional vector per image
        avg_features = patch_features.mean(dim=1)  # (B, 4)

        # Encode averaged features into a quantum circuit
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=x.device)
        for i in range(self.n_wires):
            qdev.apply(tq.RY, wires=i, params=avg_features[:, i])

        self.q_layer(qdev)
        features = self.measure(qdev).view(bsz, self.n_wires)

        return self.head(features).squeeze(-1)

__all__ = ["HybridRegressionModel", "RegressionDataset", "generate_superposition_patch_data"]
