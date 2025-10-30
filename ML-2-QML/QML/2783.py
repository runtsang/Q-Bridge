"""Quantum kernel built from a QCNN ansatz, leveraging Qiskit’s EstimatorQNN."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN


class QCNNQuantumKernel:
    """QCNN‑based quantum kernel that evaluates the inner product of ansatz states."""
    def __init__(self, n_qubits: int = 8) -> None:
        self.estimator = Estimator()
        self.n_qubits = n_qubits
        self.feature_map = ZFeatureMap(n_qubits)
        self.circuit = self._build_ansatz()
        self.qnn = EstimatorQNN(
            circuit=self.circuit.decompose(),
            observables=SparsePauliOp.from_list([("Z" + "I" * (n_qubits - 1), 1)]),
            input_params=self.feature_map.parameters,
            weight_params=self.circuit.parameters,
            estimator=self.estimator,
        )

    def _build_ansatz(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        qc.compose(self._conv_layer(self.n_qubits, "c1"), list(range(self.n_qubits)), inplace=True)
        qc.compose(self._pool_layer(list(range(self.n_qubits // 2)), list(range(self.n_qubits // 2, self.n_qubits)), "p1"),
                    list(range(self.n_qubits)), inplace=True)
        qc.compose(self._conv_layer(self.n_qubits // 2, "c2"), list(range(self.n_qubits // 2, self.n_qubits)), inplace=True)
        qc.compose(self._pool_layer([0, 1], [2, 3], "p2"),
                    list(range(self.n_qubits // 2, self.n_qubits)), inplace=True)
        qc.compose(self._conv_layer(self.n_qubits // 4, "c3"), list(range(self.n_qubits // 2, self.n_qubits)), inplace=True)
        qc.compose(self._pool_layer([0], [1], "p3"),
                    list(range(self.n_qubits // 2, self.n_qubits)), inplace=True)
        return qc

    def _conv_circuit(self, params: ParameterVector) -> QuantumCircuit:
        target = QuantumCircuit(2)
        target.rz(-np.pi / 2, 1)
        target.cx(1, 0)
        target.rz(params[0], 0)
        target.ry(params[1], 1)
        target.cx(0, 1)
        target.ry(params[2], 1)
        target.cx(1, 0)
        target.rz(np.pi / 2, 0)
        return target

    def _conv_layer(self, num_qubits: int, param_prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits)
        qubits = list(range(num_qubits))
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits * 3)
        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            qc.append(self._conv_circuit(params[param_index:param_index + 3]), [q1, q2])
            qc.barrier()
            param_index += 3
        for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
            qc.append(self._conv_circuit(params[param_index:param_index + 3]), [q1, q2])
            qc.barrier()
            param_index += 3
        return qc

    def _pool_circuit(self, params: ParameterVector) -> QuantumCircuit:
        target = QuantumCircuit(2)
        target.rz(-np.pi / 2, 1)
        target.cx(1, 0)
        target.rz(params[0], 0)
        target.ry(params[1], 1)
        target.cx(0, 1)
        target.ry(params[2], 1)
        return target

    def _pool_layer(self, sources: list[int], sinks: list[int], param_prefix: str) -> QuantumCircuit:
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits)
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
        for src, sink in zip(sources, sinks):
            qc.append(self._pool_circuit(params[param_index:param_index + 3]), [src, sink])
            qc.barrier()
            param_index += 3
        return qc

    def similarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the quantum kernel similarity between two input vectors."""
        x_np = x.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        return torch.tensor(self.qnn.evaluate([x_np, y_np])[0], dtype=torch.float32)


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    """Compute the Gram matrix between two collections of inputs using the QCNN quantum kernel."""
    qk = QCNNQuantumKernel()
    return np.array([[qk.similarity(x, y).item() for y in b] for x in a])


__all__ = ["QCNNQuantumKernel", "kernel_matrix"]
