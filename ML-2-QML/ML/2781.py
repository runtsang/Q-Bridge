"""Classical components of a hybrid quantum kernel network.

The module defines the classical RBF feature extractor and the overall
`UnifiedQuantumKernelNet` architecture that concatenates classical and
quantum kernel features before feeding them into a quantum‑parameterised
expectation head.  The quantum kernel and hybrid head are imported from
the separate quantum module `quantum_kernel_qml`.  All operations are
fully differentiable, enabling end‑to‑end training with PyTorch.
"""

import torch
import torch.nn as nn

# Import quantum kernel and hybrid expectation from the quantum module.
# The quantum module resides in the same package and implements the
# required quantum functionality.
from quantum_kernel_qml import VariationalKernel, HybridExpectation


class ClassicalRBFFeature(nn.Module):
    """Compute a vector of classical RBF kernels between an input and a
    set of reference points (the *kernel‑basis*)."""

    def __init__(self, reference_points: torch.Tensor, gamma: float = 1.0) -> None:
        super().__init__()
        # Store reference points as a buffer to avoid gradient tracking.
        self.register_buffer("reference_points", reference_points)
        self.gamma = gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, features)
        diff = x.unsqueeze(1) - self.reference_points.unsqueeze(0)  # (batch, n_ref, features)
        dist_sq = torch.sum(diff * diff, dim=2)  # (batch, n_ref)
        return torch.exp(-self.gamma * dist_sq)


class UnifiedQuantumKernelNet(nn.Module):
    """Hybrid network that concatenates classical RBF and quantum kernel
    features and maps them to a binary probability via a quantum expectation
    head."""

    def __init__(
        self,
        reference_points: torch.Tensor,
        gamma: float = 1.0,
        n_qubits: int = 4,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        self.rbf = ClassicalRBFFeature(reference_points, gamma)
        self.qkernel = VariationalKernel(n_qubits, n_layers)
        self.feature_dim = 2 * reference_points.shape[0]
        self.fc = nn.Linear(self.feature_dim, n_qubits)
        self.hybrid = HybridExpectation(n_qubits, n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rbf_feat = self.rbf(x)
        qfeat = self.qkernel(x, self.rbf.reference_points)
        combined = torch.cat([rbf_feat, qfeat], dim=1)
        mapped = self.fc(combined)
        out = self.hybrid(mapped)
        prob = torch.sigmoid(out)
        return prob
