"""
Hybrid self‑attention + EstimatorQNN implementation for quantum inference.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import Estimator as StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp

class HybridSelfAttentionEstimator:
    """
    Quantum‑classical hybrid: a Qiskit self‑attention circuit followed by a Qiskit EstimatorQNN regressor.
    """

    def __init__(self, n_qubits: int = 4, n_features: int = 2) -> None:
        self.n_qubits = n_qubits
        self.n_features = n_features
        self.attention_circuit = self._build_attention_circuit()
        self.estimator_qnn = self._build_estimator_qnn()

    def _build_attention_circuit(self) -> QuantumCircuit:
        """
        Builds a parameterised self‑attention style circuit.
        """
        qr = QuantumRegister(self.n_qubits, "q")
        cr = ClassicalRegister(self.n_qubits, "c")
        circuit = QuantumCircuit(qr, cr)

        # Rotation parameters
        rot_params = [Parameter(f"rot_{i}") for i in range(3 * self.n_qubits)]
        # Entanglement parameters
        ent_params = [Parameter(f"ent_{i}") for i in range(self.n_qubits - 1)]

        for i in range(self.n_qubits):
            circuit.rx(rot_params[3 * i], i)
            circuit.ry(rot_params[3 * i + 1], i)
            circuit.rz(rot_params[3 * i + 2], i)

        for i in range(self.n_qubits - 1):
            circuit.cnot(i, i + 1)
            circuit.rz(ent_params[i], i + 1)

        circuit.measure(qr, cr)
        return circuit

    def _build_estimator_qnn(self) -> QiskitEstimatorQNN:
        """
        Configures a Qiskit EstimatorQNN that uses the attention circuit as its feature map.
        """
        # Define a simple observable for the regression task
        observable = SparsePauliOp.from_list([("Y" * self.n_qubits, 1)])

        # Parameters: first n_features for inputs, remaining for weights
        input_params = [Parameter(f"inp_{i}") for i in range(self.n_features)]
        weight_params = [Parameter(f"w_{i}") for i in range(self.n_features)]

        estimator = StatevectorEstimator()

        return QiskitEstimatorQNN(
            circuit=self.attention_circuit,
            observables=observable,
            input_params=input_params,
            weight_params=weight_params,
            estimator=estimator,
        )

    def run(self, backend: qiskit.providers.Backend, inputs: np.ndarray, shots: int = 1024) -> dict:
        """
        Execute the hybrid circuit on the given backend.
        """
        # Map inputs to circuit parameters
        param_bindings = {}
        for i, val in enumerate(inputs.flatten()):
            param_bindings[f"inp_{i}"] = val
        # Random initial weights for demonstration
        for i in range(self.n_features):
            param_bindings[f"w_{i}"] = np.random.rand()

        job = qiskit.execute(
            self.estimator_qnn.circuit,
            backend,
            shots=shots,
            parameter_binds=[param_bindings],
        )
        result = job.result()
        return result.get_counts(self.estimator_qnn.circuit)

__all__ = ["HybridSelfAttentionEstimator"]
