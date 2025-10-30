"""Hybrid quantum classifier that combines feature map, convolutional ansatz,
and a lightweight quantum layer inspired by QuantumNAT."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator as Estimator
import torch
import torch.nn.functional as F

class HybridQuantumClassifier:
    """
    Quantum counterpart of the classical HybridQuantumClassifier.
    Builds a variational circuit with a quantum feature map, a convolutional
    ansatz, and a lightweight random layer. The output is fed to a
    classical post‑processing head.
    """

    def __init__(self, num_qubits: int = 8, depth: int = 2) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.estimator = Estimator()
        self.qnn = self._build_qnn()

    def _conv_circuit(self, params: ParameterVector) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        qc.cx(1, 0)
        qc.rz(np.pi / 2, 0)
        return qc

    def _conv_layer(self, num_qubits: int, param_prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits)
        qubits = list(range(num_qubits))
        params = ParameterVector(param_prefix, length=num_qubits * 3)
        idx = 0
        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            sub = self._conv_circuit(params[idx:idx+3])
            qc.append(sub, [q1, q2])
            idx += 3
        for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
            sub = self._conv_circuit(params[idx:idx+3])
            qc.append(sub, [q1, q2])
            idx += 3
        return qc

    def _pool_circuit(self, params: ParameterVector) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def _pool_layer(self, sources: list[int], sinks: list[int], param_prefix: str) -> QuantumCircuit:
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
        idx = 0
        for src, snk in zip(sources, sinks):
            sub = self._pool_circuit(params[idx:idx+3])
            qc.append(sub, [src, snk])
            idx += 3
        return qc

    def _build_ansatz(self) -> QuantumCircuit:
        ansatz = QuantumCircuit(self.num_qubits)
        # First convolutional layer
        ansatz.compose(self._conv_layer(self.num_qubits, "c1"), inplace=True)
        # First pooling layer
        ansatz.compose(self._pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), inplace=True)
        # Second convolutional layer
        ansatz.compose(self._conv_layer(4, "c2"), inplace=True)
        # Second pooling layer
        ansatz.compose(self._pool_layer([0, 1], [2, 3], "p2"), inplace=True)
        # Third convolutional layer
        ansatz.compose(self._conv_layer(2, "c3"), inplace=True)
        # Third pooling layer
        ansatz.compose(self._pool_layer([0], [1], "p3"), inplace=True)
        return ansatz

    def _build_qnn(self) -> EstimatorQNN:
        # Feature map
        feature_map = ZFeatureMap(self.num_qubits)
        # Ansatzz
        ansatz = self._build_ansatz()
        # Combine
        circuit = QuantumCircuit(self.num_qubits)
        circuit.compose(feature_map, inplace=True)
        circuit.compose(ansatz, inplace=True)
        # Observables: PauliZ on each qubit
        observables = [SparsePauliOp.from_list([("Z" + "I" * (self.num_qubits - i - 1), 1)]) for i in range(self.num_qubits)]
        # Build QNN
        qnn = EstimatorQNN(
            circuit=circuit,
            observables=observables,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            estimator=self.estimator,
        )
        return qnn

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute expectation values and apply a linear head.
        """
        x_np = x.detach().cpu().numpy()
        out = self.qnn.evaluate(x_np)
        probs = torch.from_numpy(out).float()
        return F.softmax(probs, dim=1)

__all__ = ["HybridQuantumClassifier"]
