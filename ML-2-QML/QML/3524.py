import torch
import torchquantum as tq
import numpy as np
from typing import Sequence
import torch.nn as nn
import torch.utils.data as data
from torchquantum.functional import func_name_dict


class QuantumAnsatz(tq.QuantumModule):
    """Data‑dependent quantum encoding using a prescribed gate list."""

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


class QuantumKernel(tq.QuantumModule):
    """Fixed quantum kernel based on a shallow Ry‑ansatz."""

    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = QuantumAnsatz(
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


def quantum_kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    """Compute Gram matrix using the quantum kernel."""
    kernel = QuantumKernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])


def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate quantum states of the form |ψ〉 = cosθ|0…0〉 + e^{iφ}sinθ|1…1〉."""
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


class RegressionDataset(data.Dataset):
    """Dataset pairing quantum states with regression targets."""

    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, index: int):
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class HybridKernelRegression(tq.QuantumModule):
    """Kernel ridge regression that uses the quantum kernel."""

    def __init__(self, num_wires: int, alpha: float = 1.0):
        super().__init__()
        self.num_wires = num_wires
        self.alpha = alpha
        self.kernel = QuantumKernel(n_wires=num_wires)
        self.w = None
        self.X_train = None

    def set_training_data(self, X: torch.Tensor) -> None:
        """Store training features for later prediction."""
        self.X_train = X.detach().clone()

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """Fit kernel ridge regression with the quantum kernel."""
        self.set_training_data(X)
        X_np = X.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        K = quantum_kernel_matrix(X_np, X_np)
        A = K + self.alpha * np.eye(K.shape[0])
        self.w = np.linalg.solve(A, y_np)

    def predict(self, X_test: torch.Tensor) -> torch.Tensor:
        """Predict outputs for new quantum samples."""
        if self.w is None or self.X_train is None:
            raise RuntimeError("Model has not been fitted.")
        X_test_np = X_test.detach().cpu().numpy()
        X_train_np = self.X_train.detach().cpu().numpy()
        K_test = quantum_kernel_matrix(X_test_np, X_train_np)
        y_pred = K_test @ self.w
        return torch.tensor(y_pred, dtype=torch.float32)


__all__ = ["QuantumAnsatz", "QuantumKernel", "quantum_kernel_matrix",
           "generate_superposition_data", "RegressionDataset", "HybridKernelRegression"]
