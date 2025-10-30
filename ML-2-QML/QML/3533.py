"""Hybrid quantum‑classical kernel and regression utilities.

This module extends the original quantum kernel implementation
with a classical RBF kernel and an ensemble regression model that
combines the strengths of both worlds.  The :class:`HybridKernel`
produces a Gram matrix that is the sum of a classical RBF component
and a pre‑computed quantum kernel matrix.  The :class:`HybridRegression`
averages predictions from a classical neural network and a quantum
variational circuit.
"""

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torchquantum.functional import func_name_dict
from typing import Sequence, Optional

# ------------------------------------------------------------------
# Quantum kernel utilities
# ------------------------------------------------------------------
class KernalAnsatz(tq.QuantumModule):
    """Encodes classical data via a list of quantum gates."""
    def __init__(self, func_list):
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

class Kernel(tq.QuantumModule):
    """Quantum kernel evaluated via a fixed TorchQuantum ansatz."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = KernalAnsatz(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    """Compute Gram matrix between datasets ``a`` and ``b``."""
    kernel = Kernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# ------------------------------------------------------------------
# Classical RBF kernel (for hybrid use)
# ------------------------------------------------------------------
class KernalAnsatzCls(nn.Module):
    """Classical RBF kernel ansatz."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class KernelCls(nn.Module):
    """Wrapper exposing the RBF kernel as a torch module."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatzCls(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

def kernel_matrix_cls(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Compute classical RBF Gram matrix."""
    kernel = KernelCls(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# ------------------------------------------------------------------
# Hybrid kernel combining quantum and classical RBF components
# ------------------------------------------------------------------
class HybridKernel:
    """Hybrid kernel that adds a classical RBF kernel to a quantum kernel."""
    def __init__(self, gamma: float = 1.0, quantum_matrix: Optional[np.ndarray] = None) -> None:
        self.gamma = gamma
        self.quantum_matrix = quantum_matrix

    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Return hybrid Gram matrix."""
        # Classical part
        x = torch.tensor(a, dtype=torch.float32)
        y = torch.tensor(b, dtype=torch.float32)
        classical = kernel_matrix_cls(x, y, self.gamma)
        # Quantum part
        if self.quantum_matrix is not None:
            return classical + self.quantum_matrix
        return classical

# ------------------------------------------------------------------
# Dataset generation (quantum superposition data)
# ------------------------------------------------------------------
def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample states of the form cos(theta)|0..0> + e^{i phi} sin(theta)|1..1>."""
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
    """Torch dataset wrapping the quantum states."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

# ------------------------------------------------------------------
# Quantum regression model
# ------------------------------------------------------------------
class QModel(tq.QuantumModule):
    """Variational quantum circuit followed by a classical head."""
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

# ------------------------------------------------------------------
# Classical regression model (for ensemble)
# ------------------------------------------------------------------
class ClassicalQModel(nn.Module):
    """Simple feed‑forward regression network (classical counterpart)."""
    def __init__(self, num_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

# ------------------------------------------------------------------
# Hybrid regression as an ensemble of classical and quantum models
# ------------------------------------------------------------------
class HybridRegression:
    """Ensemble of a classical and a quantum regression model."""
    def __init__(self, num_features: int, num_wires: int):
        self.classical = ClassicalQModel(num_features)
        self.quantum = QModel(num_wires)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, lr: float = 1e-3):
        """Train both models independently and keep their parameters."""
        # Classical fit
        optimizer_c = torch.optim.Adam(self.classical.parameters(), lr=lr)
        criterion = nn.MSELoss()
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        for _ in range(epochs):
            optimizer_c.zero_grad()
            pred_c = self.classical(X_t)
            loss_c = criterion(pred_c, y_t)
            loss_c.backward()
            optimizer_c.step()

        # Quantum fit
        optimizer_q = torch.optim.Adam(self.quantum.parameters(), lr=lr)
        for _ in range(epochs):
            optimizer_q.zero_grad()
            pred_q = self.quantum(X_t)
            loss_q = criterion(pred_q, y_t)
            loss_q.backward()
            optimizer_q.step()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the average of classical and quantum predictions."""
        X_t = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            pred_c = self.classical(X_t)
            pred_q = self.quantum(X_t)
        return ((pred_c + pred_q).cpu().numpy())

__all__ = [
    "KernalAnsatz",
    "Kernel",
    "kernel_matrix",
    "HybridKernel",
    "generate_superposition_data",
    "RegressionDataset",
    "QModel",
    "ClassicalQModel",
    "HybridRegression",
]
