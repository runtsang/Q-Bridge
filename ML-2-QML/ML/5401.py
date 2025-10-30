"""Unified kernel method with classical RBF kernel and optional convolution."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

class ConvFilter(nn.Module):
    """Simple 2‑D convolution filter used as a drop‑in replacement for a quantum filter."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def run(self, data: np.ndarray) -> float:
        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean().item()

class ClassicalRBF(nn.Module):
    """Classical radial basis function kernel."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class QuantumKernelMethod(nn.Module):
    """Unified kernel method that can use a classical RBF kernel or a quantum kernel."""
    def __init__(self, *, use_quantum: bool = False, gamma: float = 1.0,
                 kernel_size: int = 2, threshold: float = 0.0,
                 device: str | torch.device = 'cpu') -> None:
        super().__init__()
        self.use_quantum = use_quantum
        self.gamma = gamma
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.device = device

        if use_quantum:
            # The quantum kernel implementation is provided in the qml module.
            self.kernel = None
        else:
            self.kernel = ClassicalRBF(gamma)

        self.conv = ConvFilter(kernel_size, threshold)

    def _extract_features(self, X: np.ndarray) -> torch.Tensor:
        """Apply the convolution filter to each sample and return a 2‑D tensor."""
        feats = []
        for sample in X:
            feats.append(self.conv.run(sample))
        return torch.tensor(feats, dtype=torch.float32, device=self.device).unsqueeze(1)

    def compute_kernel_matrix(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Return the Gram matrix between X and Y."""
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        Y_t = torch.tensor(Y, dtype=torch.float32, device=self.device)
        if self.use_quantum:
            raise NotImplementedError("Quantum kernel should be used from the qml module.")
        K = self.kernel(X_t, Y_t).cpu().numpy()
        return K

    def predict(self, X_train: np.ndarray, y_train: np.ndarray,
                X_test: np.ndarray, alpha: float = 1e-3) -> np.ndarray:
        """Kernel ridge regression prediction."""
        K = self.compute_kernel_matrix(X_train, X_train)
        n = K.shape[0]
        K_reg = K + alpha * np.eye(n, dtype=K.dtype)
        coeffs = np.linalg.solve(K_reg, y_train)
        K_test = self.compute_kernel_matrix(X_test, X_train)
        return K_test @ coeffs

__all__ = ["QuantumKernelMethod", "ClassicalRBF", "ConvFilter"]
