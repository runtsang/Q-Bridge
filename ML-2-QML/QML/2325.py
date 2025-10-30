"""Quantum implementation of QCNNGen129 combining convolutional layers
and a sampler subcircuit.

The circuit follows the same structure as the classical model:
feature map → conv layers → pool layers → sampler measurement.
It returns a Qiskit EstimatorQNN for classification and a SamplerQNN
for probability distribution over two outcomes.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit_machine_learning.neural_networks import EstimatorQNN, SamplerQNN
from qiskit_machine_learning.utils import algorithm_globals

class QCNNGen129Quantum:
    """
    Quantum hybrid model that mirrors the classical QCNNGen129 architecture.
    It exposes two outputs:
        - classification: expectation value of Z on qubit 0
        - sampler: probability distribution over |0⟩/|1⟩ on qubit 1
    """

    def __init__(self) -> None:
        algorithm_globals.random_seed = 12345
        self.estimator = StatevectorEstimator()
        self.sampler = StatevectorSampler()
        # Build feature map and ansatz
        self.feature_map = ZFeatureMap(8)
        self.circuit = self._build_ansatz()
        # Observation for classification
        self.classification_obs = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

    def _conv_circuit(self, params: ParameterVector) -> QuantumCircuit:
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

    def _pool_circuit(self, params: ParameterVector) -> QuantumCircuit:
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
            sub = self._conv_circuit(params[idx:idx+3])
            qc.append(sub, [q1, q2])
            idx += 3
        return qc

    def _pool_layer(self, sources: list[int], sinks: list[int], param_prefix: str) -> QuantumCircuit:
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(param_prefix, length=len(sources) * 3)
        idx = 0
        for s, t in zip(sources, sinks):
            sub = self._pool_circuit(params[idx:idx+3])
            qc.append(sub, [s, t])
            idx += 3
        return qc

    def _build_ansatz(self) -> QuantumCircuit:
        ansatz = QuantumCircuit(8, name="Ansatz")
        # First conv + pool
        ansatz.compose(self._conv_layer(8, "c1"), inplace=True)
        ansatz.compose(self._pool_layer([0,1,2,3], [4,5,6,7], "p1"), inplace=True)
        # Second conv + pool
        ansatz.compose(self._conv_layer(4, "c2"), inplace=True)
        ansatz.compose(self._pool_layer([0,1], [2,3], "p2"), inplace=True)
        # Third conv + pool
        ansatz.compose(self._conv_layer(2, "c3"), inplace=True)
        ansatz.compose(self._pool_layer([0], [1], "p3"), inplace=True)
        return ansatz

    def _build_sampler(self) -> QuantumCircuit:
        """Sampler subcircuit that produces a 2‑outcome distribution."""
        qc = QuantumCircuit(2)
        inputs = ParameterVector("x", 2)
        weights = ParameterVector("w", 4)
        qc.ry(inputs[0], 0)
        qc.ry(inputs[1], 1)
        qc.cx(0, 1)
        qc.ry(weights[0], 0)
        qc.ry(weights[1], 1)
        qc.cx(0, 1)
        qc.ry(weights[2], 0)
        qc.ry(weights[3], 1)
        return qc

    def build_qnn(self) -> EstimatorQNN:
        """Return the classification QNN."""
        circuit = QuantumCircuit(8)
        circuit.compose(self.feature_map, inplace=True)
        circuit.compose(self.circuit, inplace=True)
        qnn = EstimatorQNN(
            circuit=circuit.decompose(),
            observables=self.classification_obs,
            input_params=self.feature_map.parameters,
            weight_params=self.circuit.parameters,
            estimator=self.estimator,
        )
        return qnn

    def build_sampler_qnn(self) -> SamplerQNN:
        """Return the sampler QNN."""
        sampler_circuit = self._build_sampler()
        inputs = ParameterVector("x", 2)
        weights = ParameterVector("w", 4)
        sampler_qnn = SamplerQNN(
            circuit=sampler_circuit,
            input_params=inputs,
            weight_params=weights,
            sampler=self.sampler,
        )
        return sampler_qnn

    def forward(self, inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Run both the classification QNN and the sampler QNN.
        `inputs` should be an array of shape (n_samples, 8) for the feature map
        and (n_samples, 2) for the sampler inputs. The method returns a tuple
        (classification, sampler_distribution).
        """
        qnn = self.build_qnn()
        sampler_qnn = self.build_sampler_qnn()
        # Classification prediction
        class_pred = qnn.predict(inputs)
        # Sampler prediction
        sampler_pred = sampler_qnn.predict(inputs[:, :2])  # use first 2 dims as sampler inputs
        return class_pred, sampler_pred

def QCNN() -> QCNNGen129Quantum:
    """Factory returning the hybrid QCNNGen129Quantum model."""
    return QCNNGen129Quantum()

__all__ = ["QCNN", "QCNNGen129Quantum"]
