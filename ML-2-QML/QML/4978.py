"""Quantum kernel framework using TorchQuantum with hybrid support.

This module mirrors the classical version but replaces the RBF kernel with a
parameterised variational circuit.  It also exposes the same builder
utilities for Qiskit classifiers, photonic fraud detection and data
generation, making the quantum and classical sides share a common API.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from typing import Iterable, Tuple, Any
import torchquantum as tq

# ----------------------------------------------------------------------
# Classical RBF kernel (used in hybrid mode)
# ----------------------------------------------------------------------
class ClassicalKernel(nn.Module):
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

# ----------------------------------------------------------------------
# Quantum kernel implementation
# ----------------------------------------------------------------------
class QuantumKernelImpl(tq.QuantumModule):
    def __init__(self, n_wires: int = 4, depth: int = 2):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict[f"{n_wires}xRy"]
        )
        self.random_layer = tq.RandomLayer(
            n_ops=depth * 10, wires=list(range(self.n_wires))
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, qdev: tq.QuantumDevice) -> Tensor:
        self.encoder(qdev)
        self.random_layer(qdev)
        return torch.abs(self.measure(qdev).view(-1)[0])

# ----------------------------------------------------------------------
# Unified quantum kernel method
# ----------------------------------------------------------------------
class QuantumKernelMethod(tq.QuantumModule):
    """
    Hybrid kernel that can operate purely quantum, purely classical or a
    geometric mean of both.  The interface mirrors the classical class.
    """

    def __init__(
        self,
        mode: str = "quantum",
        gamma: float = 1.0,
        n_wires: int = 4,
        depth: int = 2,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.mode = mode
        self.gamma = gamma
        self.n_wires = n_wires
        self.depth = depth
        self.device = device

        if mode == "classical":
            self.kernel = ClassicalKernel(gamma)
        elif mode == "quantum":
            self.kernel = QuantumKernelImpl(n_wires, depth)
        else:  # hybrid
            self.classical = ClassicalKernel(gamma)
            self.quantum = QuantumKernelImpl(n_wires, depth)

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
    # Kernel ridge regression
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
    # Utility builders
    # ------------------------------------------------------------------
    @staticmethod
    def build_classifier_circuit(num_qubits: int, depth: int):
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

        def _clip(v, bound): return max(-bound, min(bound, v))

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
    def generate_superposition_data(num_wires: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
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

    @staticmethod
    def RegressionDataset(samples: int, num_wires: int):
        from torch.utils.data import Dataset

        class RegressionDataset(Dataset):
            def __init__(self):
                self.states, self.labels = QuantumKernelMethod.generate_superposition_data(num_wires, samples)

            def __len__(self): return len(self.states)

            def __getitem__(self, idx):
                return {
                    "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
                    "target": torch.tensor(self.labels[idx], dtype=torch.float32),
                }

        return RegressionDataset()

__all__ = ["QuantumKernelMethod"]
