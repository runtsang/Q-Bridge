"""Unified quantum sampler that couples a feature‑map, convolution‑pooling ansatz, and an attention‑style entanglement block.

The circuit is wrapped in an EstimatorQNN and exposed via a class with a run method that accepts a classical input vector."""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator, StatevectorSampler
from qiskit_machine_learning.neural_networks import EstimatorQNN

class UnifiedSamplerQNN:
    """
    Variational quantum sampler that mirrors the classical architecture:
    - Feature map
    - Convolution + pooling ansatz
    - Attention‑style entanglement
    - State‑vector sampling
    """
    def __init__(self, n_qubits: int = 8) -> None:
        self.n_qubits = n_qubits
        self._build_circuit()

    def _conv_layer(self, num_qubits: int, prefix: str) -> QuantumCircuit:
        """Construct a convolution layer with 3‑parameter per pair."""
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(f"{prefix}_c", length=num_qubits * 3)
        for i in range(0, num_qubits, 2):
            qc.append(self._conv_circuit(params[i:i+3]), [i, i+1])
        return qc

    def _conv_circuit(self, params: ParameterVector) -> QuantumCircuit:
        """Single 2‑qubit convolution block."""
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

    def _pool_layer(self, num_qubits: int, prefix: str) -> QuantumCircuit:
        """Pooling layer that discards half the qubits."""
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(f"{prefix}_p", length=num_qubits // 2 * 3)
        for i in range(0, num_qubits, 2):
            qc.append(self._pool_circuit(params[i:i+3]), [i, i+1])
        return qc

    def _pool_circuit(self, params: ParameterVector) -> QuantumCircuit:
        """Single 2‑qubit pooling block."""
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def _attention_circuit(self, rotation: ParameterVector, entangle: ParameterVector) -> QuantumCircuit:
        """Attention‑style entanglement using RX/RY/RZ rotations and controlled‑RX."""
        qc = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            qc.rx(rotation[3*i], i)
            qc.ry(rotation[3*i + 1], i)
            qc.rz(rotation[3*i + 2], i)
        for i in range(self.n_qubits - 1):
            qc.crx(entangle[i], i, i + 1)
        return qc

    def _build_circuit(self) -> None:
        # Feature map
        self.feature_map = ZFeatureMap(self.n_qubits)
        # Ansatz layers
        ansatz = QuantumCircuit(self.n_qubits)
        ansatz.compose(self._conv_layer(self.n_qubits, "c1"), inplace=True)
        ansatz.compose(self._pool_layer(self.n_qubits, "p1"), inplace=True)
        ansatz.compose(self._conv_layer(self.n_qubits // 2, "c2"), inplace=True)
        ansatz.compose(self._pool_layer(self.n_qubits // 2, "p2"), inplace=True)
        ansatz.compose(self._conv_layer(self.n_qubits // 4, "c3"), inplace=True)
        ansatz.compose(self._pool_layer(self.n_qubits // 4, "p3"), inplace=True)

        # Attention parameters
        rotation_params = ParameterVector("rot", length=self.n_qubits * 3)
        entangle_params = ParameterVector("ent", length=self.n_qubits - 1)
        attention = self._attention_circuit(rotation_params, entangle_params)

        # Combine all
        circuit = QuantumCircuit(self.n_qubits)
        circuit.compose(self.feature_map, inplace=True)
        circuit.compose(ansatz, inplace=True)
        circuit.compose(attention, inplace=True)

        self.circuit = circuit.decompose()

        # Observable
        observable = SparsePauliOp.from_list([("Z" + "I" * (self.n_qubits - 1), 1)])

        # Estimator and sampler
        estimator = Estimator()
        self.sampler = StatevectorSampler()
        self.qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=observable,
            input_params=self.feature_map.parameters,
            weight_params=ansatz.parameters + rotation_params + entangle_params,
            estimator=estimator,
        )

    def run(self, inputs: np.ndarray, shots: int = 1024) -> np.ndarray:
        """
        Run the quantum sampler on a batch of inputs and return the probability of the
        observable.  Inputs are expected to be in the same shape as the feature map.
        """
        probs = self.qnn.predict(inputs, shots=shots)
        return probs

def SamplerQNN() -> UnifiedSamplerQNN:
    """Factory returning a fully‑configured :class:`UnifiedSamplerQNN`."""
    return UnifiedSamplerQNN()
