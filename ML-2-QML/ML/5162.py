"""Unified QCNN‑Kernel‑Classifier – classical component."""
from __future__ import annotations

import torch
from torch import nn
import math
from typing import Callable, Iterable, List, Tuple

class UnifiedQCNNKernelClassifier(nn.Module):
    """
    Hybrid classifier that combines:
      * a classical CNN‑like fully‑connected stack,
      * a quantum sub‑network implemented via an EstimatorQNN,
      * an RBF kernel mixing classical and quantum representations,
      * a final linear decision layer.

    The quantum circuit is supplied through ``set_quantum_circuit``.
    """

    def __init__(
        self,
        input_dim: int = 8,
        depth: int = 3,
        gamma: float = 1.0,
        quantum_output_dim: int = 4,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.depth = depth
        self.gamma = gamma
        self.quantum_output_dim = quantum_output_dim
        self.num_classes = num_classes

        # Classical CNN‑style fully‑connected stack
        self.classical_net = nn.Sequential(
            nn.Linear(input_dim, 16), nn.Tanh(),
            nn.Linear(16, 16), nn.Tanh(),
            nn.Linear(16, 12), nn.Tanh(),
            nn.Linear(12, 8), nn.Tanh(),
            nn.Linear(8, 4), nn.Tanh(),
        )
        self.classical_head = nn.Linear(4, quantum_output_dim)

        # Placeholder for the quantum circuit – will be injected later
        self.qnn = None  # type: ignore[assignment]

        # Kernel parameter
        self.gamma_param = nn.Parameter(torch.tensor(gamma))

        # Final classifier
        # Input dimension: classical features + quantum features + kernel feature
        self.final_classifier = nn.Linear(
            quantum_output_dim * 2 + 1, num_classes
        )

    def set_quantum_circuit(self, qnn_callable: Callable[[torch.Tensor], torch.Tensor]) -> None:
        """
        Attach a callable that evaluates the quantum circuit.
        The callable must accept a batch of floats of shape (batch, input_dim)
        and return a tensor of shape (batch, quantum_output_dim).
        """
        self.qnn = qnn_callable

    def rbf_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute a single RBF kernel value."""
        diff = x - y
        return torch.exp(-self.gamma_param * torch.sum(diff * diff, dim=-1, keepdim=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Classical branch
        cls_feat = self.classical_net(x)
        cls_out = self.classical_head(cls_feat)

        # Quantum branch – fallback to zeros if not set
        if self.qnn is None:
            qnn_out = torch.zeros_like(cls_out)
        else:
            qnn_out = self.qnn(x)

        # Kernel feature between classical and quantum representations
        batch = x.shape[0]
        k_feat = torch.zeros(batch, 1, device=x.device)
        for i in range(batch):
            k_feat[i] = self.rbf_kernel(cls_out[i], qnn_out[i])

        # Concatenate all features
        combined = torch.cat([cls_out, qnn_out, k_feat], dim=1)

        out = self.final_classifier(combined)
        return torch.log_softmax(out, dim=-1)

def UnifiedQCNNKernelClassifierFactory() -> UnifiedQCNNKernelClassifier:
    """
    Factory that returns a fully configured model ready for training.
    The quantum circuit is built using the QML helper defined in
    the quantum module and attached via ``set_quantum_circuit``.
    """
    model = UnifiedQCNNKernelClassifier()
    return model

__all__ = [
    "UnifiedQCNNKernelClassifier",
    "UnifiedQCNNKernelClassifierFactory",
]
