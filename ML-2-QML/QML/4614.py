"""Hybrid quantum QCNN implementation combining convolutional layers,
a random “quanv” filter, and an EstimatorQNN for regression.

The circuit re‑uses the QCNN ansatz from the reference but inserts a
random circuit (QuanvCircuit) on the first four qubits to emulate a
quantum convolutional filter. The resulting circuit is wrapped into
an EstimatorQNN instance that can be evaluated on classical data.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

class HybridQCNN:
    """
    Hybrid quantum QCNN.

    Parameters
    ----------
    backend : str, optional
        Backend name for the StatevectorEstimator. Defaults to ``qasm_simulator``.
    shots : int, optional
        Number of shots for the state‑vector estimator. Defaults to 1024.
    """

    def __init__(self, backend: str = "qasm_simulator", shots: int = 1024) -> None:
        self.backend = backend
        self.shots = shots

        # Feature map and ansatz construction
        self.feature_map = ZFeatureMap(8)
        self.ansatz = self._build_ansatz()
        self.observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

        # EstimatorQNN wrapper
        self.estimator = StatevectorEstimator()
        self.qnn = EstimatorQNN(
            circuit=self.ansatz,
            observables=self.observable,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            estimator=self.estimator,
        )

    # ----------------------------------------------------------------------- #
    # Circuit construction helpers
    # ----------------------------------------------------------------------- #
    def _conv_circuit(self, params: ParameterVector) -> QuantumCircuit:
        """Single‑pair convolution gate set."""
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
        """Convolutional sub‑circuit operating on qubit pairs."""
        qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
        qubits = list(range(num_qubits))
        params = ParameterVector(param_prefix, length=num_qubits * 3)
        param_index = 0
        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            qc.compose(self._conv_circuit(params[param_index:param_index + 3]), [q1, q2])
            qc.barrier()
            param_index += 3
        for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
            qc.compose(self._conv_circuit(params[param_index:param_index + 3]), [q1, q2])
            qc.barrier()
            param_index += 3
        return qc

    def _pool_circuit(self, params: ParameterVector) -> QuantumCircuit:
        """Single‑pair pooling gate set."""
        target = QuantumCircuit(2)
        target.rz(-np.pi / 2, 1)
        target.cx(1, 0)
        target.rz(params[0], 0)
        target.ry(params[1], 1)
        target.cx(0, 1)
        target.ry(params[2], 1)
        return target

    def _pool_layer(self, sources: list[int], sinks: list[int], param_prefix: str) -> QuantumCircuit:
        """Pooling sub‑circuit that merges qubits."""
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits, name="Pooling Layer")
        params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
        param_index = 0
        for source, sink in zip(sources, sinks):
            qc.compose(self._pool_circuit(params[param_index:param_index + 3]), [source, sink])
            qc.barrier()
            param_index += 3
        return qc

    def _quanv_circuit(self, n_qubits: int) -> QuantumCircuit:
        """Random circuit acting as a quantum convolution filter."""
        qc = QuantumCircuit(n_qubits)
        for i in range(n_qubits):
            qc.rx(np.pi / 2, i)
        qc += random_circuit(n_qubits, 2)
        return qc

    def _build_ansatz(self) -> QuantumCircuit:
        """Assemble the full QCNN ansatz with an inserted Quanv filter."""
        ansatz = QuantumCircuit(8)

        # Convolutional Layer 1
        conv1 = self._conv_layer(8, "c1")
        ansatz.compose(conv1, list(range(8)), inplace=True)

        # Insert a random Quanv filter on the first four qubits
        quanv = self._quanv_circuit(4)
        ansatz.compose(quanv, [0, 1, 2, 3], inplace=True)

        # Pooling Layer 1
        pool1 = self._pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1")
        ansatz.compose(pool1, list(range(8)), inplace=True)

        # Convolutional Layer 2
        conv2 = self._conv_layer(4, "c2")
        ansatz.compose(conv2, list(range(4, 8)), inplace=True)

        # Pooling Layer 2
        pool2 = self._pool_layer([0, 1], [2, 3], "p2")
        ansatz.compose(pool2, list(range(4, 8)), inplace=True)

        # Convolutional Layer 3
        conv3 = self._conv_layer(2, "c3")
        ansatz.compose(conv3, list(range(6, 8)), inplace=True)

        # Pooling Layer 3
        pool3 = self._pool_layer([0], [1], "p3")
        ansatz.compose(pool3, list(range(6, 8)), inplace=True)

        return ansatz.decompose()

    # ----------------------------------------------------------------------- #
    # Public API
    # ----------------------------------------------------------------------- #
    def run(self, data: np.ndarray) -> np.ndarray:
        """
        Evaluate the hybrid QCNN on classical data.

        Parameters
        ----------
        data : np.ndarray
            Input array of shape (batch, 8) representing 8‑dimensional feature vectors.

        Returns
        -------
        np.ndarray
            Prediction array of shape (batch, 1).
        """
        return self.qnn.predict(data)

__all__ = ["HybridQCNN"]
