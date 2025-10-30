from __future__ import annotations

import torch
from torch import nn
import numpy as np
from typing import Iterable, Sequence

class Conv__gen438(nn.Module):
    """
    Classical hybrid convolutional filter that emulates the original Conv
    filter but adds a variational quantum layer and fraud‑regularisation.
    """
    def __init__(
        self,
        kernel_size: int = 2,
        conv_out_channels: int = 8,
        quantum_threshold: float = 0.0,
        fraud_params: dict | None = None,
        gamma: float = 1.0,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size

        # 1. Classical convolutional backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(1, conv_out_channels, kernel_size, bias=True),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        # 2. Classical emulation of the quantum variational filter
        self.quantum_filter = _QuantumFilter(conv_out_channels, quantum_threshold)

        # 3. Fraud‑regularisation block
        if fraud_params is None:
            fraud_params = {
                "bs_theta": 1.0,
                "bs_phi": 0.0,
                "phases": (0.0, 0.0),
                "squeeze_r": (0.0, 0.0),
                "squeeze_phi": (0.0, 0.0),
                "displacement_r": (1.0, 1.0),
                "displacement_phi": (0.0, 0.0),
                "shift": (0.0, 0.0),
            }
        self.fraud_block = _FraudRegBlock(fraud_params)

        # 4. RBF kernel regulariser
        self.kernel = _RBFKernel(gamma)

    def run(self, data: np.ndarray) -> float:
        """
        Run the hybrid filter on a 2‑D array of shape (kernel_size, kernel_size).

        Parameters
        ----------
        data : np.ndarray
            Input image patch.

        Returns
        -------
        float
            Scalar output of the filter.
        """
        # Convert to torch tensor
        x = torch.as_tensor(data, dtype=torch.float32).view(1, 1, self.kernel_size, self.kernel_size)

        # Feature extraction
        feat = self.backbone(x)                      # (1, C, 1, 1)
        feat_flat = feat.view(1, feat.size(1))       # (1, C)

        # Quantum filter emulation
        q_out = self.quantum_filter(feat_flat)       # (1, 1)

        # Duplicate for fraud block (expects 2‑dim input)
        q_out_expanded = torch.cat([q_out, q_out], dim=1)  # (1, 2)

        # Fraud regularisation
        fraud_out = self.fraud_block(q_out_expanded)          # (1, 2)

        # Collapse to a single scalar
        return fraud_out.mean().item()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(kernel_size={self.kernel_size})"

# Helper classes
class _QuantumFilter(nn.Module):
    """
    Classical emulation of a quantum variational filter.
    Uses a learnable parameter vector to model the effect of a
    variational circuit.
    """
    def __init__(self, n_qubits: int, threshold: float = 0.0) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.threshold = threshold
        self.thetas = nn.Parameter(torch.randn(n_qubits))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(self.thetas * x + self.threshold)
        return probs.mean(dim=1, keepdim=True)

class _RBFKernel(nn.Module):
    """
    Computes a pairwise RBF kernel matrix between two feature sets.
    """
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        diff = a.unsqueeze(1) - b.unsqueeze(0)
        squared = (diff ** 2).sum(-1)
        return torch.exp(-self.gamma * squared)

class _FraudRegBlock(nn.Module):
    """
    Small linear‑activation‑scale‑shift block that mimics the photonic
    fraud‑detector structure.
    """
    def __init__(self, params: dict) -> None:
        super().__init__()
        weight = torch.tensor([[params["bs_theta"], params["bs_phi"]],
                               [params["squeeze_r"][0], params["squeeze_r"][1]]],
                               dtype=torch.float32)
        bias = torch.tensor(params["phases"], dtype=torch.float32)

        self.linear = nn.Linear(2, 2)
        with torch.no_grad():
            self.linear.weight.copy_(weight)
            self.linear.bias.copy_(bias)

        self.activation = nn.Tanh()
        self.register_buffer("scale", torch.tensor(params["displacement_r"], dtype=torch.float32))
        self.register_buffer("shift", torch.tensor(params["displacement_phi"], dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.linear(x)
        y = self.activation(y)
        y = y * self.scale + self.shift
        return y
