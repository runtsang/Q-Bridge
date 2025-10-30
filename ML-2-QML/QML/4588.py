from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data for quantum regression."""
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0
    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels

class RegressionDataset(torch.utils.data.Dataset):
    """Dataset for quantum regression."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class RBFKernelLayer(nn.Module):
    """Classical RBF kernel feature map used downstream of the quantum circuit."""
    def __init__(self, num_centroids: int, dim: int, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.centroids = nn.Parameter(torch.randn(num_centroids, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        diff = x.unsqueeze(1) - self.centroids.unsqueeze(0)
        dist_sq = (diff * diff).sum(dim=-1)
        return torch.exp(-self.gamma * dist_sq)

class HybridRegressionModel(tq.QuantumModule):
    """Quantum regression model that combines a variational circuit with a classical RBF kernel."""
    class QLayer(tq.QuantumModule):
        """Variational layer with a random circuit and parameterised single‑qubit gates."""
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            self.crx = tq.CRX(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            for w in range(self.n_wires):
                self.rx(qdev, wires=w)
                self.ry(qdev, wires=w)
                self.rz(qdev, wires=w)
                self.crx(qdev, wires=[w, (w + 1) % self.n_wires])
            tqf.hadamard(qdev, wires=list(range(self.n_wires)))
            tqf.sx(qdev, wires=list(range(self.n_wires)))
            for w in range(self.n_wires):
                tqf.cnot(qdev, wires=[(w + 1) % self.n_wires, w])

    def __init__(self,
                 num_wires: int = 4,
                 num_centroids: int = 8,
                 gamma: float = 1.0):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self.QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.kernel_layer = RBFKernelLayer(num_centroids, num_wires, gamma)
        self.head = nn.Linear(num_centroids, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)  # (bsz, n_wires)
        k_features = self.kernel_layer(features)
        out = self.head(k_features)
        return out.squeeze(-1)

def kernel_matrix(a: list[torch.Tensor],
                  b: list[torch.Tensor],
                  gamma: float = 1.0) -> np.ndarray:
    """Compute Gram matrix using the same RBF kernel as in HybridRegressionModel."""
    a_tensor = torch.stack(a)
    b_tensor = torch.stack(b)
    diff = a_tensor.unsqueeze(1) - b_tensor.unsqueeze(0)
    dist_sq = (diff * diff).sum(dim=-1)
    return torch.exp(-gamma * dist_sq).numpy()

__all__ = ["generate_superposition_data",
           "RegressionDataset",
           "HybridRegressionModel",
           "RBFKernelLayer"]
