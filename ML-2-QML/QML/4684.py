"""Quantum implementation of the hybrid QCNN model.

The circuit is built from a feature map followed by three
convolution‑pool blocks and an optional output layer that
performs either regression (expectation value) or classification
(sampling distribution).  The architecture mirrors the
classical version defined in ``QCNN__gen131.py`` but uses
Qiskit primitives for the variational part.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, ParameterVector, algorithm_globals
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit_machine_learning.neural_networks import EstimatorQNN, SamplerQNN


class QCNNHybrid:
    """Quantum implementation of the hybrid QCNN model."""

    def __init__(self, num_qubits: int = 8, seed: int | None = None) -> None:
        self.num_qubits = num_qubits
        if seed is not None:
            algorithm_globals.random_seed = seed
        self.feature_map = ZFeatureMap(num_qubits)
        self.ansatz = self._build_ansatz()

    @staticmethod
    def _conv_circuit(params: ParameterVector) -> QuantumCircuit:
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
    def _pool_circuit(params: ParameterVector) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def _conv_layer(self, num_qubits: int, param_prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
        idx = 0
        for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
            conv = self._conv_circuit(params[idx: idx + 3])
            qc.append(conv, [q1, q2])
            qc.barrier()
            idx += 3
        return qc

    def _pool_layer(
        self, sources: list[int], sinks: list[int], param_prefix: str
    ) -> QuantumCircuit:
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(param_prefix, length=len(sources) * 3)
        idx = 0
        for src, snk in zip(sources, sinks):
            pool = self._pool_circuit(params[idx: idx + 3])
            qc.append(pool, [src, snk])
            qc.barrier()
            idx += 3
        return qc

    def _build_ansatz(self) -> QuantumCircuit:
        ansatz = QuantumCircuit(self.num_qubits)
        ansatz.compose(self._conv_layer(self.num_qubits, "c1"), inplace=True)
        ansatz.compose(
            self._pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"),
            inplace=True,
        )
        ansatz.compose(self._conv_layer(self.num_qubits // 2, "c2"), inplace=True)
        ansatz.compose(
            self._pool_layer([0, 1], [2, 3], "p2"),
            inplace=True,
        )
        ansatz.compose(self._conv_layer(self.num_qubits // 4, "c3"), inplace=True)
        ansatz.compose(
            self._pool_layer([0], [1], "p3"),
            inplace=True,
        )
        return ansatz

    def get_qnn(self, mode: str = "regression") -> EstimatorQNN | SamplerQNN:
        """Return a Qiskit QNN object for the requested mode.

        Parameters
        ----------
        mode : str, optional
            Either ``'regression'`` or ``'classification'``.
            Defaults to ``'regression'``.
        """
        circuit = QuantumCircuit(self.num_qubits)
        circuit.compose(self.feature_map, range(self.num_qubits), inplace=True)
        circuit.compose(self.ansatz, range(self.num_qubits), inplace=True)

        if mode == "regression":
            observable = SparsePauliOp.from_list([("Y" * self.num_qubits, 1)])
            estimator = StatevectorEstimator()
            return EstimatorQNN(
                circuit=circuit.decompose(),
                observables=observable,
                input_params=self.feature_map.parameters,
                weight_params=self.ansatz.parameters,
                estimator=estimator,
            )
        elif mode == "classification":
            sampler = StatevectorSampler()
            return SamplerQNN(
                circuit=circuit.decompose(),
                input_params=self.feature_map.parameters,
                weight_params=self.ansatz.parameters,
                sampler=sampler,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

__all__ = ["QCNNHybrid"]
