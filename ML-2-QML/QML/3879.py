"""Quantum QCNN with efficient estimator and shot‑noise emulation."""

from __future__ import annotations

import numpy as np
from typing import Iterable, List, Sequence

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators import SparsePauliOp
from qiskit.primitives import Estimator as StateEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals

class QCNNGen187:
    """Quantum circuit implementation of the QCNN with shot‑noise support."""
    def __init__(self, num_qubits: int = 8, seed: int | None = None) -> None:
        algorithm_globals.random_seed = seed or 12345
        self.num_qubits = num_qubits
        self.estimator = StateEstimator()
        self.circuit = self._build_circuit()

    # ---------- Helper layers ----------
    def _conv_circuit(self, params: Sequence[float]) -> QuantumCircuit:
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

    def _pool_circuit(self, params: Sequence[float]) -> QuantumCircuit:
        target = QuantumCircuit(2)
        target.rz(-np.pi / 2, 1)
        target.cx(1, 0)
        target.rz(params[0], 0)
        target.ry(params[1], 1)
        target.cx(0, 1)
        target.ry(params[2], 1)
        return target

    def _conv_layer(self, num_qubits: int, prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits)
        for i in range(0, num_qubits, 2):
            params = ParameterVector(f"{prefix}_q{i}", 3)
            sub = self._conv_circuit(params)
            qc.compose(sub, [i, (i + 1) % num_qubits], inplace=True)
            qc.barrier()
        return qc

    def _pool_layer(self, sources: Sequence[int], sinks: Sequence[int], prefix: str) -> QuantumCircuit:
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits)
        for src, sink in zip(sources, sinks):
            params = ParameterVector(f"{prefix}_s{src}_t{sink}", 3)
            sub = self._pool_circuit(params)
            qc.compose(sub, [src, sink], inplace=True)
            qc.barrier()
        return qc

    # ---------- Circuit construction ----------
    def _build_circuit(self) -> QuantumCircuit:
        feature_map = ZFeatureMap(self.num_qubits)
        ansatz = QuantumCircuit(self.num_qubits, name="Ansatz")

        # First convolution and pooling
        ansatz.compose(self._conv_layer(self.num_qubits, "c1"), inplace=True)
        ansatz.compose(self._pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), inplace=True)

        # Second convolution and pooling on the reduced register
        ansatz.compose(self._conv_layer(self.num_qubits // 2, "c2"), inplace=True)
        ansatz.compose(self._pool_layer([0, 1], [2, 3], "p2"), inplace=True)

        # Third convolution and pooling on the smallest register
        ansatz.compose(self._conv_layer(self.num_qubits // 4, "c3"), inplace=True)
        ansatz.compose(self._pool_layer([0], [1], "p3"), inplace=True)

        circuit = QuantumCircuit(self.num_qubits)
        circuit.compose(feature_map, inplace=True)
        circuit.compose(ansatz, inplace=True)
        return circuit.decompose()

    def to_qnn(self) -> EstimatorQNN:
        observable = SparsePauliOp.from_list([("Z" + "I" * (self.num_qubits - 1), 1)])
        weight_params = [p for p in self.circuit.parameters if "c" in p.name]
        return EstimatorQNN(
            circuit=self.circuit,
            observables=observable,
            input_params=self.circuit.parameters,
            weight_params=weight_params,
            estimator=self.estimator,
        )

    # ---------- Evaluation ----------
    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """Evaluate expectation values, optionally adding Gaussian shot noise."""
        qnn = self.to_qnn()
        raw = qnn.evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in raw:
            noisy_row = [rng.normal(mean, max(1e-6, 1 / shots)) for mean in row]
            noisy.append(noisy_row)
        return noisy

def create_qcnn_gen187(num_qubits: int = 8, seed: int | None = None) -> QCNNGen187:
    """Factory returning a QCNNGen187 instance."""
    return QCNNGen187(num_qubits, seed)

__all__ = ["QCNNGen187", "create_qcnn_gen187"]
