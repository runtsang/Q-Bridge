"""Unified kernel method with quantum kernel implementation."""
from __future__ import annotations

import numpy as np
import torch
import torchquantum as tq

class QuantumKernelMethod(tq.QuantumModule):
    """Unified kernel method that uses a quantum kernel."""
    def __init__(self, *, n_wires: int = 4,
                 gate_list: list | None = None,
                 device: str | torch.device = 'cpu') -> None:
        super().__init__()
        self.n_wires = n_wires
        self.device = device
        self.q_device = tq.QuantumDevice(n_wires=n_wires, bsz=1, device=device)

        if gate_list is None:
            gate_list = [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        self.gate_list = gate_list
        self.ansatz = self._build_ansatz(gate_list)

    def _build_ansatz(self, gate_list: list) -> tq.QuantumModule:
        class KernalAnsatz(tq.QuantumModule):
            def __init__(self, func_list):
                super().__init__()
                self.func_list = func_list

            @tq.static_support
            def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
                q_device.reset_states(x.shape[0])
                for info in self.func_list:
                    params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
                    tq.func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
                for info in reversed(self.func_list):
                    params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
                    tq.func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        return KernalAnsatz(gate_list)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

    def compute_kernel_matrix(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Return the Gram matrix between X and Y using the quantum kernel."""
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        Y_t = torch.tensor(Y, dtype=torch.float32, device=self.device)
        K = self.forward(X_t, Y_t).cpu().numpy()
        return K

    def predict(self, X_train: np.ndarray, y_train: np.ndarray,
                X_test: np.ndarray, alpha: float = 1e-3) -> np.ndarray:
        """Kernel ridge regression prediction using the quantum kernel."""
        K = self.compute_kernel_matrix(X_train, X_train)
        n = K.shape[0]
        K_reg = K + alpha * np.eye(n, dtype=K.dtype)
        coeffs = np.linalg.solve(K_reg, y_train)
        K_test = self.compute_kernel_matrix(X_test, X_train)
        return K_test @ coeffs

__all__ = ["QuantumKernelMethod"]
