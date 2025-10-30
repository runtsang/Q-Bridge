"""Hybrid quantum kernel method with classical and quantum components.

The class supports three modes:
* ``classical`` – pure RBF kernel,
* ``quantum`` – pure TorchQuantum variational kernel,
* ``hybrid`` – geometric mean of the two.
It also provides utilities to build Qiskit classifiers, StrawberryFields
fraud‑detection programs, and regression datasets, and a lightweight
kernel‑ridge regression interface.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from typing import Iterable, Tuple, Any

# ----------------------------------------------------------------------
# Classical RBF kernel
# ----------------------------------------------------------------------
class ClassicalKernel(nn.Module):
    """Pure RBF kernel implemented with PyTorch."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

# ----------------------------------------------------------------------
# Quantum kernel (TorchQuantum)
# ----------------------------------------------------------------------
class QuantumKernel(nn.Module):
    """Variational quantum kernel based on TorchQuantum."""
    def __init__(self, n_wires: int = 4, depth: int = 2):
        super().__init__()
        import torchquantum as tq
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{n_wires}xRy"])
        self.random_layer = tq.RandomLayer(n_ops=depth * 10, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        bsz = 1  # compute kernel for a single pair
        self.q_device.reset_states(bsz)
        self.encoder(self.q_device, x)
        self.random_layer(self.q_device)
        self.encoder(self.q_device, -y)
        return torch.abs(self.measure(self.q_device).view(-1)[0])

# ----------------------------------------------------------------------
# Unified kernel method
# ----------------------------------------------------------------------
class QuantumKernelMethod(nn.Module):
    """
    Unified kernel framework.

    Parameters
    ----------
    mode:
        ``classical`` – only RBF kernel,
        ``quantum`` – only TorchQuantum kernel,
        ``hybrid`` – geometric mean of the two.
    gamma:
        RBF width parameter.
    n_wires:
        Number of qubits for the quantum kernel.
    depth:
        Depth of the variational layer.
    """
    def __init__(
        self,
        mode: str = "classical",
        gamma: float = 1.0,
        n_wires: int = 4,
        depth: int = 2,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.mode = mode
        self.device = device
        self.gamma = gamma
        self.n_wires = n_wires
        self.depth = depth

        if mode == "classical":
            self.kernel = ClassicalKernel(gamma).to(device)
        elif mode == "quantum":
            self.kernel = QuantumKernel(n_wires, depth).to(device)
        elif mode == "hybrid":
            self.classical = ClassicalKernel(gamma).to(device)
            self.quantum = QuantumKernel(n_wires, depth).to(device)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        # placeholders for regression
        self.alpha: Tensor | None = None
        self.X_train: Tensor | None = None

    # ------------------------------------------------------------------
    # Kernel matrix
    # ------------------------------------------------------------------
    def kernel_matrix(self, a: Tensor, b: Tensor) -> Tensor:
        if self.mode == "classical":
            return self._classical_matrix(a, b)
        elif self.mode == "quantum":
            return self._quantum_matrix(a, b)
        else:
            return self._hybrid_matrix(a, b)

    def _classical_matrix(self, a: Tensor, b: Tensor) -> Tensor:
        return torch.exp(-self.gamma * torch.cdist(a, b, p=2) ** 2)

    def _quantum_matrix(self, a: Tensor, b: Tensor) -> Tensor:
        n_a, n_b = a.shape[0], b.shape[0]
        out = torch.empty((n_a, n_b), device=self.device)
        for i in range(n_a):
            for j in range(n_b):
                out[i, j] = self.kernel(a[i].unsqueeze(0), b[j].unsqueeze(0))
        return out

    def _hybrid_matrix(self, a: Tensor, b: Tensor) -> Tensor:
        return torch.sqrt(self._classical_matrix(a, b) * self._quantum_matrix(a, b))

    # ------------------------------------------------------------------
    # Simple kernel ridge regression
    # ------------------------------------------------------------------
    def fit_ridge(self, X: Tensor, y: Tensor, alpha: float = 1.0):
        K = self.kernel_matrix(X, X)
        n = K.shape[0]
        I = torch.eye(n, device=self.device)
        self.alpha = torch.linalg.solve(K + alpha * I, y)
        self.X_train = X

    def predict(self, X: Tensor) -> Tensor:
        if self.alpha is None:
            raise RuntimeError("Model must be fitted before prediction.")
        K = self.kernel_matrix(X, self.X_train)
        return K @ self.alpha

    # ------------------------------------------------------------------
    # Utility builders (classifier, fraud detection, regression)
    # ------------------------------------------------------------------
    @staticmethod
    def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[Any, Iterable[Any], Iterable[Any], list[Any]]:
        from qiskit import QuantumCircuit
        from qiskit.circuit import ParameterVector
        from qiskit.quantum_info import SparsePauliOp

        encoding = ParameterVector("x", num_qubits)
        weights = ParameterVector("theta", num_qubits * depth)

        qc = QuantumCircuit(num_qubits)
        for param, qubit in zip(encoding, range(num_qubits)):
            qc.rx(param, qubit)

        idx = 0
        for _ in range(depth):
            for qubit in range(num_qubits):
                qc.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(num_qubits - 1):
                qc.cz(qubit, qubit + 1)

        observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
            for i in range(num_qubits)
        ]
        return qc, list(encoding), list(weights), observables

    @staticmethod
    def build_fraud_detection_program(input_params, layers):
        from strawberryfields import Program
        from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate

        def _clip(v, bound):
            return max(-bound, min(bound, v))

        def _apply_layer(modes, params, clip):
            BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
            for i, phase in enumerate(params.phases):
                Rgate(phase) | modes[i]
            for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
                Sgate(r if not clip else _clip(r, 5), phi) | modes[i]
            BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
            for i, phase in enumerate(params.phases):
                Rgate(phase) | modes[i]
            for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
                Dgate(r if not clip else _clip(r, 5), phi) | modes[i]
            for i, k in enumerate(params.kerr):
                Kgate(k if not clip else _clip(k, 1)) | modes[i]

        prog = Program(2)
        with prog.context as q:
            _apply_layer(q, input_params, clip=False)
            for layer in layers:
                _apply_layer(q, layer, clip=True)
        return prog

    @staticmethod
    def generate_superposition_data(num_features: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
        x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
        angles = x.sum(axis=1)
        y = np.sin(angles) + 0.1 * np.cos(2 * angles)
        return x, y.astype(np.float32)

    @staticmethod
    def RegressionDataset(samples: int, num_features: int):
        from torch.utils.data import Dataset

        class RegressionDataset(Dataset):
            def __init__(self):
                self.features, self.labels = QuantumKernelMethod.generate_superposition_data(num_features, samples)

            def __len__(self):
                return len(self.features)

            def __getitem__(self, idx):
                return {
                    "states": torch.tensor(self.features[idx], dtype=torch.float32),
                    "target": torch.tensor(self.labels[idx], dtype=torch.float32),
                }

        return RegressionDataset()

# ----------------------------------------------------------------------
# Backwards‑compatibility wrappers
# ----------------------------------------------------------------------
class KernalAnsatz(ClassicalKernel):
    def __init__(self, gamma: float = 1.0):
        super().__init__(gamma)

class Kernel(ClassicalKernel):
    def __init__(self, gamma: float = 1.0):
        super().__init__(gamma)

def kernel_matrix(a, b, gamma=1.0):
    return QuantumKernelMethod(mode="classical", gamma=gamma).kernel_matrix(
        torch.tensor(a), torch.tensor(b)
    )

__all__ = [
    "QuantumKernelMethod",
    "KernalAnsatz",
    "Kernel",
    "kernel_matrix",
]
