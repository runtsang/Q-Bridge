"""Quantum hybrid sampler circuit combining QCNN layers and a statevector sampler."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator, StatevectorSampler
from qiskit_machine_learning.neural_networks import EstimatorQNN

class QuantumHybridSampler:
    """Quantum counterpart of the hybrid sampler.
    Builds a QCNN‑style variational circuit, then exposes two interfaces:
    1) a classification expectation value via EstimatorQNN.
    2) a 2‑class sampling probability using StatevectorSampler on qubit 0."""
    
    def __init__(self, seed: int | None = None) -> None:
        self.estimator = Estimator()
        self.sampler = StatevectorSampler()
        if seed is not None:
            self.estimator.random_seed = seed
            self.sampler.random_seed = seed
        self.circuit, self.feature_map, self.ansatz = self._build_circuit()
        # Define observable for classification: Pauli Z on qubit 0
        self.observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])
        self.qnn = EstimatorQNN(
            circuit=self.circuit.decompose(),
            observables=self.observable,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            estimator=self.estimator,
        )
    
    # --------------------------------------------------------------------------- #
    # Helper circuits
    # --------------------------------------------------------------------------- #
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
    
    def _conv_layer(self, num_qubits: int, prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
        qubits = list(range(num_qubits))
        param_index = 0
        params = ParameterVector(prefix, length=num_qubits * 3)
        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            sub = self._conv_circuit(params[param_index : param_index + 3])
            qc.compose(sub, [q1, q2], inplace=True)
            qc.barrier()
            param_index += 3
        for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
            sub = self._conv_circuit(params[param_index : param_index + 3])
            qc.compose(sub, [q1, q2], inplace=True)
            qc.barrier()
            param_index += 3
        return qc
    
    def _pool_layer(self, sources: list[int], sinks: list[int], prefix: str) -> QuantumCircuit:
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits, name="Pooling Layer")
        param_index = 0
        params = ParameterVector(prefix, length=(num_qubits // 2) * 3)
        for s, t in zip(sources, sinks):
            sub = self._pool_circuit(params[param_index : param_index + 3])
            qc.compose(sub, [s, t], inplace=True)
            qc.barrier()
            param_index += 3
        return qc
    
    def _build_circuit(self) -> tuple[QuantumCircuit, ZFeatureMap, QuantumCircuit]:
        feature_map = ZFeatureMap(8)
        ansatz = QuantumCircuit(8, name="Ansatz")
        # First Convolutional Layer
        ansatz.compose(self._conv_layer(8, "c1"), range(8), inplace=True)
        # First Pooling Layer
        ansatz.compose(self._pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), range(8), inplace=True)
        # Second Convolutional Layer
        ansatz.compose(self._conv_layer(4, "c2"), range(4, 8), inplace=True)
        # Second Pooling Layer
        ansatz.compose(self._pool_layer([0, 1], [2, 3], "p2"), range(4, 8), inplace=True)
        # Third Convolutional Layer
        ansatz.compose(self._conv_layer(2, "c3"), range(6, 8), inplace=True)
        # Third Pooling Layer
        ansatz.compose(self._pool_layer([0], [1], "p3"), range(6, 8), inplace=True)
        # Combine feature map and ansatz
        circuit = QuantumCircuit(8)
        circuit.compose(feature_map, range(8), inplace=True)
        circuit.compose(ansatz, range(8), inplace=True)
        return circuit, feature_map, ansatz
    
    # --------------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------------- #
    def sample(self, input_params: list[float], weight_params: list[float], shots: int = 1024) -> np.ndarray:
        """Return a 2‑element array of sampling probabilities for qubit 0."""
        bind_dict = {p: v for p, v in zip(self.feature_map.parameters + self.ansatz.parameters, input_params + weight_params)}
        bound_circ = self.circuit.bind_parameters(bind_dict)
        result = self.sampler.run(bound_circ, shots=shots)[0]
        # Extract probabilities of |0> and |1> on qubit 0
        probs = np.zeros(2)
        for state, prob in result.items():
            if state[0] == '0':
                probs[0] += prob
            else:
                probs[1] += prob
        return probs
    
    def classify(self, input_params: list[float], weight_params: list[float]) -> float:
        """Return the expectation value of Z on qubit 0, serving as a binary classifier."""
        bind_dict = {p: v for p, v in zip(self.feature_map.parameters + self.ansatz.parameters, input_params + weight_params)}
        exp_vals = self.qnn.run(bind_dict)[0]
        return float(exp_vals)
    
    def forward(self, input_params: list[float], weight_params: list[float]) -> tuple[float, np.ndarray]:
        """Convenience wrapper returning (classification, sampler distribution)."""
        return self.classify(input_params, weight_params), self.sample(input_params, weight_params)

def SamplerQNN() -> QuantumHybridSampler:
    """Factory returning a quantum instance compatible with the classical SamplerQNN interface."""
    return QuantumHybridSampler()

__all__ = ["QuantumHybridSampler", "SamplerQNN"]
