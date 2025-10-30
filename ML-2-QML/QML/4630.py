"""QuantumKernelMethod - Quantum hybrid interface.

This module mirrors the classical version but replaces the kernel and
neural‑network components with their Qiskit/TorchQuantum counterparts.
The API is kept identical so experiments can switch freely between the
two backends.
"""

from __future__ import annotations

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.optimizers import COBYLA
from typing import Sequence

__all__ = ["QuantumKernelMethod"]


# --------------------------------------------------------------------------- #
# Quantum kernels
# --------------------------------------------------------------------------- #
class KernalAnsatz(tq.QuantumModule):
    """Encodes classical data via a list of parametrised gates."""

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
    """TorchQuantum RBF‑style kernel with a simple Ry ansatz."""

    def __init__(self, n_wires: int = 4, gamma: float = 1.0) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.gamma = gamma
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = self._build_ansatz()

    def _build_ansatz(self):
        func_list = [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(self.n_wires)]
        return KernalAnsatz(func_list)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])


# --------------------------------------------------------------------------- #
# Quantum EstimatorQNN
# --------------------------------------------------------------------------- #
class QuantumEstimatorQNN:
    """Lightweight 1‑qubit EstimatorQNN inspired by the Qiskit example."""

    def __init__(self):
        params = [Parameter("in"), Parameter("w")]
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.ry(params[0], 0)
        qc.rx(params[1], 0)
        self.circuit = qc
        obs = SparsePauliOp.from_list([("Y", 1)])
        self.estimator = StatevectorEstimator()
        self.qnn = EstimatorQNN(
            circuit=qc,
            observables=obs,
            input_params=[params[0]],
            weight_params=[params[1]],
            estimator=self.estimator,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # Simple COBYLA optimisation of the weight parameter
        def objective(w):
            self.qnn.set_weight_params([w[0]])
            preds = np.array(
                [self.qnn.forward(torch.tensor([x], dtype=torch.float32))[0].item()
                 for x in X]
            )
            return np.mean((preds - y) ** 2)

        opt = COBYLA(maxiter=200)
        # COBYLA requires an initial guess; we use 0.0
        opt.optimize([0.0], objective)  # placeholder; real use would vary


# --------------------------------------------------------------------------- #
# Quantum QCNN helper
# --------------------------------------------------------------------------- #
class QuantumQCNN:
    """QCNN circuit builder with conv and pool layers."""

    def __init__(self, n_qubits: int = 8):
        self.n_qubits = n_qubits
        self.circuit = self._build_circuit()

    @staticmethod
    def _conv_circuit(params):
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        qc.cx(1, 0)
        qc.rz(np.pi / 2, 0)
        return qc

    @staticmethod
    def _pool_circuit(params):
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def _conv_layer(self, num_qubits, param_prefix):
        qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
        qubits = list(range(num_qubits))
        params = ParameterVector(param_prefix, length=num_qubits * 3)
        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            qc.compose(self._conv_circuit(params[q1:q1+3]), [q1, q2], inplace=True)
            qc.barrier()
        return qc

    def _pool_layer(self, sources, sinks, param_prefix):
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits, name="Pooling Layer")
        params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
        for src, snk in zip(sources, sinks):
            qc.compose(self._pool_circuit(params[src:src+3]), [src, snk], inplace=True)
            qc.barrier()
        return qc

    def _build_circuit(self):
        # Feature map
        fm = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            fm.h(i)
        # Assemble layers
        qc = QuantumCircuit(self.n_qubits)
        qc.append(self._conv_layer(self.n_qubits, "c1"), range(self.n_qubits))
        qc.append(self._pool_layer(list(range(0, 4)), list(range(4, 8)), "p1"), range(self.n_qubits))
        qc.append(self._conv_layer(4, "c2"), range(4, 8))
        qc.append(self._pool_layer([0, 1], [2, 3], "p2"), range(4, 8))
        qc.append(self._conv_layer(2, "c3"), range(6, 8))
        qc.append(self._pool_layer([0], [1], "p3"), range(6, 8))
        # Combine feature map and ansatz
        full = QuantumCircuit(self.n_qubits)
        full.append(fm, range(self.n_qubits))
        full.append(qc, range(self.n_qubits))
        return full.decompose()

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Placeholder: no real prediction, returns zeros
        return np.zeros(len(X))


# --------------------------------------------------------------------------- #
# Unified hybrid class
# --------------------------------------------------------------------------- #
class QuantumKernelMethod:
    """Quantum hybrid interface mirroring the classical version.

    Parameters
    ----------
    mode: {"kernel", "qnn", "qcnn"}
        Backend to use.  ``kernel`` returns a quantum kernel, ``qnn`` uses
        the 1‑qubit EstimatorQNN, and ``qcnn`` uses the QCNN circuit.
    n_wires: int, optional
        Number of wires for the quantum kernel (default: 4).
    """

    def __init__(self, mode: str = "kernel", n_wires: int = 4) -> None:
        self.mode = mode
        if mode == "kernel":
            self.kernel = QuantumKernel(n_wires=n_wires)
        elif mode == "qnn":
            self.model = QuantumEstimatorQNN()
        elif mode == "qcnn":
            self.model = QuantumQCNN(n_qubits=8)
        else:
            raise ValueError(f"Unsupported mode {mode}")

    # --------------------------------------------------------------------- #
    # Kernel utilities
    # --------------------------------------------------------------------- #
    def compute_kernel_matrix(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        if self.mode!= "kernel":
            raise RuntimeError("Kernel matrix only available in 'kernel' mode")
        return np.array(
            [
                [
                    self.kernel(
                        torch.tensor([x], dtype=torch.float32),
                        torch.tensor([y], dtype=torch.float32),
                    ).item()
                    for y in Y
                ]
                for x in X
            ]
        )

    # --------------------------------------------------------------------- #
    # Training & prediction
    # --------------------------------------------------------------------- #
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if self.mode == "kernel":
            # No trainable parameters in this simple kernel
            pass
        elif self.mode == "qnn":
            self.model.fit(X, y)
        elif self.mode == "qcnn":
            # Placeholder – real QCNN training would use a quantum optimiser
            pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.mode == "kernel":
            raise RuntimeError("Prediction not defined for kernel mode")
        elif self.mode == "qnn":
            return np.array(
                [
                    self.model.qnn.forward(torch.tensor([x], dtype=torch.float32))[0].item()
                    for x in X
                ]
            )
        elif self.mode == "qcnn":
            return self.model.predict(X)
        else:
            raise RuntimeError("Unsupported mode")
