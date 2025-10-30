"""Hybrid kernel regression framework with quantum kernel and variational regression."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torchquantum.functional import func_name_dict
from torchquantum import op_name_dict


def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample quantum states of the form cos(theta)|0...0> + e^{i phi} sin(theta)|1...1>.
    The labels are a non-linear function of the angles, providing a challenging regression task.
    """
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
    """
    Dataset wrapper for quantum regression data.
    Each sample contains a complex amplitude vector and a real target.
    """
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class KernalAnsatz(tq.QuantumModule):
    """Quantum kernel ansatz that encodes two classical vectors via a reversible circuit."""
    def __init__(self, func_list):
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = x[:, info["input_idx"]] if op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)


class Kernel(tq.QuantumModule):
    """Quantum kernel evaluated by a fixed TorchQuantum ansatz."""
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
    """Compute the Gram matrix between two sets of samples using the quantum kernel."""
    kernel = Kernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])


class HybridKernelRegression(tq.QuantumModule):
    """
    Quantum kernel ridge regression.
    Uses a quantum kernel to compute the Gram matrix and solves the linear system
    classically. A variational circuit is added as a lightweight quantum head
    to capture residuals after the kernel step.
    """
    class QLayer(tq.QuantumModule):
        def __init__(self, num_wires: int):
            super().__init__()
            self.n_wires = num_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)

    def __init__(self, num_wires: int, alpha: float = 1.0):
        super().__init__()
        self.n_wires = num_wires
        self.alpha = alpha
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self.QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)
        self.train_X = None
        self.train_y = None
        self.w = None

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """
        Fit the kernel ridge regression model by computing the quantum kernel
        Gram matrix and solving the regularized linear system.
        """
        self.train_X = X.detach().clone()
        self.train_y = y.detach().clone()
        K = kernel_matrix(X, X)
        K += self.alpha * np.eye(K.shape[0])
        self.w = np.linalg.solve(K, self.train_y.numpy())

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict targets for new samples by evaluating the quantum kernel against
        the training data and applying the learned weights. The variational head
        refines the prediction.
        """
        if self.w is None:
            raise RuntimeError("Model has not been fitted yet.")
        K = kernel_matrix(X, self.train_X)
        preds = K @ self.w
        # Variational refinement
        bsz = X.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=X.device)
        self.encoder(qdev, X)
        self.q_layer(qdev)
        features = self.measure(qdev)
        refine = self.head(features).squeeze(-1)
        return torch.tensor(preds, dtype=torch.float32) + refine

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.predict(X)


__all__ = [
    "KernalAnsatz",
    "Kernel",
    "kernel_matrix",
    "RegressionDataset",
    "HybridKernelRegression",
]
