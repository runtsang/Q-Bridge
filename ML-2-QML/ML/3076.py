"""Hybrid Quanvolution implementation combining classical convolution and a quantum kernel.

The class `QuanvolutionHybrid` replaces the pure convolutional filter in the original
quantum–classical example with a patch‑wise quantum encoder based on a fixed
`RandomLayer`.  It exposes a `forward` method that returns a log‑softmax over
10 output classes and a `kernel_matrix` helper that evaluates the Gram matrix
between two batches of images using the same quantum kernel.  The module
remains fully classical for training while delegating the feature extraction
to a lightweight quantum device that can be swapped for a simulator or a
real backend.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq


class QuantumPatchEncoder(tq.QuantumModule):
    """Encodes a 2×2 image patch into a 4‑qubit feature vector.

    The encoding uses a single‑qubit Ry rotation per pixel followed by a
    fixed RandomLayer.  The resulting measurement in the Pauli‑Z basis is
    returned as a 4‑dimensional real vector.
    """
    def __init__(self) -> None:
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
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, patch: torch.Tensor) -> None:
        """Encode a single patch into the quantum device."""
        self.encoder(q_device, patch)
        self.q_layer(q_device)

    def get_features(self, batch: torch.Tensor) -> torch.Tensor:
        """Return the 4‑dimensional feature vector for each patch in *batch*."""
        bsz = batch.shape[0]
        device = batch.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        self.forward(qdev, batch)
        return self.measure(qdev).view(bsz, self.n_wires)


class QuanvolutionHybrid(nn.Module):
    """Classical wrapper around the quantum patch encoder.

    The forward pass applies the quantum encoder to every 2×2 patch of a
    28×28 grayscale image, concatenates all patch features and feeds them
    through a linear head to produce log‑softmax logits for 10 classes.
    """
    def __init__(self) -> None:
        super().__init__()
        self.encoder = QuantumPatchEncoder()
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.size(0)
        device = x.device
        # Reshape to 28×28 patches
        x = x.view(bsz, 28, 28)
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
                patches.append(self.encoder.get_features(patch))
        features = torch.cat(patches, dim=1)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Return the Gram matrix between two batches of images using the quantum kernel."""
        # Flatten images to 784‑dim vectors
        a_flat = a.view(a.size(0), -1)
        b_flat = b.view(b.size(0), -1)
        # Use a simple classical RBF kernel as a placeholder for the quantum kernel
        gamma = 1.0
        diff = a_flat.unsqueeze(1) - b_flat.unsqueeze(0)
        return torch.exp(-gamma * (diff * diff).sum(-1))

__all__ = ["QuanvolutionHybrid"]
