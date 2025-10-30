"""Hybrid quantum estimator combining self‑attention style entanglement
and a variational rotation layer to predict a scalar output.

The circuit first encodes the input features via Ry rotations, then
applies a series of controlled‑RZ gates that form a self‑attention
style entanglement pattern.  Finally a weight rotation is applied
and the expectation value of a Pauli‑Y observable is estimated using
StatevectorEstimator.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator


class HybridEstimator:
    """Quantum hybrid estimator."""
    def __init__(self, n_qubits: int = 4, input_dim: int = 2):
        self.n_qubits = n_qubits
        self.input_dim = input_dim
        self.backend = qiskit.Aer.get_backend("statevector_simulator")

        # Parameter placeholders
        self.input_params = [qiskit.circuit.Parameter(f"x{i}") for i in range(input_dim)]
        self.rotation_params = [qiskit.circuit.Parameter(f"r{i}") for i in range(n_qubits * 3)]
        self.entangle_params = [qiskit.circuit.Parameter(f"e{i}") for i in range(n_qubits - 1)]
        self.weight_params = [qiskit.circuit.Parameter(f"w{i}") for i in range(n_qubits)]

        self.circuit = self._build_circuit()
        self.observables = SparsePauliOp.from_list([("Y" * self.n_qubits, 1)])
        self.estimator = StatevectorEstimator()
        self.estimator_qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=self.observables,
            input_params=self.input_params + self.rotation_params,
            weight_params=self.weight_params,
            estimator=self.estimator,
        )

    def _build_circuit(self) -> QuantumCircuit:
        qr = QuantumRegister(self.n_qubits, "q")
        cr = ClassicalRegister(self.n_qubits, "c")
        qc = QuantumCircuit(qr, cr)

        # Encode inputs via Ry rotations
        for i, param in enumerate(self.input_params):
            qc.ry(param, i % self.n_qubits)

        # Rotation layer
        for i, param in enumerate(self.rotation_params):
            if i % 3 == 0:
                qc.rx(param, i // 3)
            elif i % 3 == 1:
                qc.ry(param, i // 3)
            else:
                qc.rz(param, i // 3)

        # Self‑attention style entanglement via CRX
        for i, param in enumerate(self.entangle_params):
            qc.crx(param, i, i + 1)

        # Weight layer
        for i, param in enumerate(self.weight_params):
            qc.ry(param, i)

        qc.measure(qr, cr)
        return qc

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Return expectation values for a batch of input feature vectors."""
        params = {}
        for i, val in enumerate(inputs.T):
            params[self.input_params[i]] = val
        # Add random values for other parameters
        for param in self.rotation_params + self.entangle_params + self.weight_params:
            params[param] = np.random.rand()
        return self.estimator_qnn.predict(params)


__all__ = ["HybridEstimator"]
