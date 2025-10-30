from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq
from torchquantum.functional import func_name_dict
import numpy as np
from typing import Sequence
import networkx as nx
import itertools

# ---------------------------------------------------------------------------#
#  Quantum kernel (fixed ansatz)
# ---------------------------------------------------------------------------#
class QuantumKernel(tq.QuantumModule):
    """Fixed quantum kernel that mirrors the ML kernel but uses a parameter‑free ansatz."""
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        # Simple encoding: separate Ry rotations followed by a CNOT chain
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{n_wires}xRy"])

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # broadcast to (1, d)
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        # encode x
        self.encoder(self.q_device, x)
        # apply CNOT chain
        for idx in range(self.n_wires - 1):
            tq.CNOT(self.q_device, wires=[idx, idx + 1])
        # reset and encode y with negative parameters
        self.q_device.reset_states(1)
        self.encoder(self.q_device, -y)
        # compute overlap
        return torch.abs(self.q_device.states.view(-1)[0])

def quantum_kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], n_wires: int = 4) -> np.ndarray:
    """Compute the Gram matrix using the fixed quantum kernel."""
    kernel = QuantumKernel(n_wires)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# ---------------------------------------------------------------------------#
#  Quantum regression model (quantum kernel ridge regression)
# ---------------------------------------------------------------------------#
class QuantumKRRModel(tq.QuantumModule):
    """Quantum kernel ridge regression using the fixed ansatz."""
    def __init__(self, n_wires: int = 4, alpha: float = 1.0):
        super().__init__()
        self.n_wires = n_wires
        self.alpha = alpha
        self.alpha_vec = None

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        K = torch.tensor(quantum_kernel_matrix(X, X, self.n_wires), dtype=torch.float32)
        self.alpha_vec = torch.linalg.solve(K + self.alpha * torch.eye(K.shape[0]), y)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.alpha_vec is None:
            raise RuntimeError("Model has not been fitted yet.")
        K = torch.tensor(quantum_kernel_matrix(X, X, self.n_wires), dtype=torch.float32)
        return torch.mv(K, self.alpha_vec)

# ---------------------------------------------------------------------------#
#  Hybrid quantum model: classical RBF features + quantum kernel embeddings
# ---------------------------------------------------------------------------#
class HybridQuantumModel(tq.QuantumModule):
    """Combines classical RBF features with quantum kernel embeddings."""
    def __init__(self, num_features: int, n_wires: int = 4, num_centers: int = 50, gamma: float = 1.0):
        super().__init__()
        self.num_features = num_features
        self.n_wires = n_wires
        self.num_centers = num_centers
        self.gamma = gamma
        self.register_buffer("centers", torch.rand(num_centers, num_features))
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{n_wires}xRy"])
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_centers, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        # Classical RBF features
        rbf_feats = torch.exp(-self.gamma * torch.sum((x.unsqueeze(1) - self.centers) ** 2, dim=-1))
        # Quantum features
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=batch, device=x.device)
        self.encoder(qdev, x)
        for idx in range(self.n_wires - 1):
            tq.CNOT(qdev, wires=[idx, idx + 1])
        quantum_feats = self.measure(qdev).squeeze(-1)
        # concatenate and head
        feats = torch.cat([rbf_feats, quantum_feats], dim=-1)
        return self.head(feats).squeeze(-1)

# ---------------------------------------------------------------------------#
#  Legacy quantum model (for backward compatibility)
# ---------------------------------------------------------------------------#
class QModel(tq.QuantumModule):
    """Legacy quantum regression model from the seed."""
    class QLayer(tq.QuantumModule):
        def __init__(self, num_wires: int):
            super().__init__()
            self.n_wires = num_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)

    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self.QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

__all__ = [
    "QuantumKernel",
    "quantum_kernel_matrix",
    "QuantumKRRModel",
    "HybridQuantumModel",
    "QModel",
]
