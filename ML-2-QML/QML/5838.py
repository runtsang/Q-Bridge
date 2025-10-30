import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
from torchquantum.functional import func_name_dict
from typing import Sequence

# ------------------------------------------------------------------
# Quantum kernel based on a fixed ansatz
# ------------------------------------------------------------------
class QuantumKernel(tq.QuantumModule):
    """
    Quantum kernel that encodes two data vectors into a circuit and returns
    the absolute value of the overlap of the resulting states.
    """
    def __init__(self, n_wires: int = 4, n_layers: int = 2) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        # Simple ansatz: rotate each qubit and apply a random layer
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(self.n_wires)]
        )
        self.random_layer = tq.RandomLayer(n_ops=n_layers, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the quantum kernel value for two data vectors."""
        bsz = 1  # kernel is evaluated pairwise
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=x.device)
        # Encode x
        self.encoder(qdev, x.unsqueeze(0))
        # Apply random layer
        self.random_layer(qdev)
        # Encode -y
        self.encoder(qdev, -y.unsqueeze(0))
        # Overlap measurement
        overlap = self.measure(qdev)
        return torch.abs(overlap.view(-1)[0])

    def matrix(self, X: Sequence[torch.Tensor], Y: Sequence[torch.Tensor]) -> np.ndarray:
        """Return Gram matrix between two datasets."""
        return np.array([[self.forward(x, y).item() for y in Y] for x in X])

# ------------------------------------------------------------------
# Quantum quanvolution filter
# ------------------------------------------------------------------
class QuantumQuanvolutionFilter(tq.QuantumModule):
    """
    Apply a quantum kernel to all 2×2 patches of a grayscale image.
    """
    def __init__(self, n_wires: int = 4, n_layers: int = 8) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(self.n_wires)]
        )
        self.random_layer = tq.RandomLayer(n_ops=n_layers, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, 1, 28, 28) grayscale images.
        Returns flattened quantum‑encoded feature map of shape (batch, 4*14*14).
        """
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                # Gather 2×2 patch
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
                self.random_layer(qdev)
                meas = self.measure(qdev)
                patches.append(meas.view(bsz, 4))
        return torch.cat(patches, dim=1)

class QuantumQuanvolutionClassifier(nn.Module):
    """
    Hybrid classifier that uses the quantum quanvolution filter followed by a linear head.
    """
    def __init__(self, num_classes: int = 10, in_features: int = 4 * 14 * 14) -> None:
        super().__init__()
        self.qfilter = QuantumQuanvolutionFilter()
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

# ------------------------------------------------------------------
# Unified quantum kernel + quanvolution architecture
# ------------------------------------------------------------------
class UnifiedKernelQuanvolution(nn.Module):
    """
    Combines a quantum kernel, a quantum quanvolution filter, and a linear classifier.
    The kernel can be used for kernel‑based learning, while the filter provides
    convolutional quantum features. The classifier operates on the quantum
    feature map. Users can optionally inject a custom quantum kernel.
    """
    def __init__(self,
                 n_wires: int = 4,
                 n_layers_kernel: int = 2,
                 n_layers_filter: int = 8,
                 num_classes: int = 10,
                 in_features: int = 4 * 14 * 14) -> None:
        super().__init__()
        self.quantum_kernel = QuantumKernel(n_wires=n_wires, n_layers=n_layers_kernel)
        self.qfilter = QuantumQuanvolutionFilter(n_wires=n_wires, n_layers=n_layers_filter)
        self.linear = nn.Linear(in_features, num_classes)

    def kernel_matrix(self, X: Sequence[torch.Tensor], Y: Sequence[torch.Tensor]) -> np.ndarray:
        """Return the quantum kernel Gram matrix between two datasets."""
        return self.quantum_kernel.matrix(X, Y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the quantum quanvolution classifier on input images."""
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuantumKernel", "QuantumQuanvolutionFilter", "QuantumQuanvolutionClassifier", "UnifiedKernelQuanvolution"]
