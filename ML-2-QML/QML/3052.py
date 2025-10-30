"""Quantum Self‑Attention based on a QCNN ansatz.

The circuit encodes a 4‑qubit token representation via a ZFeatureMap
and then applies a layered QCNN ansatz (convolution + pooling).
Rotation and entanglement parameters control the variational degrees
of freedom.  The measurement yields a probability distribution that
serves as the attention weights.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

class HybridSelfAttention:
    """Quantum self‑attention using a QCNN‑style ansatz."""
    def __init__(self, n_qubits: int = 4) -> None:
        self.n_qubits = n_qubits
        self.backend = qiskit.Aer.get_backend("qasm_simulator")
        self.circuit = self._build_circuit()
        self.estimator = Estimator()
        self.qnn = EstimatorQNN(
            circuit=self.circuit.decompose(),
            observables=SparsePauliOp.from_list([("Z" + "I" * (n_qubits - 1), 1)]),
            input_params=[],
            weight_params=[],
            estimator=self.estimator,
        )

    def _conv_layer(self, params: ParameterVector) -> QuantumCircuit:
        """Convolutional layer for two qubits."""
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

    def _pool_layer(self, params: ParameterVector) -> QuantumCircuit:
        """Pooling layer for two qubits."""
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def _build_circuit(self) -> QuantumCircuit:
        """Assemble a 4‑qubit QCNN ansatz."""
        qc = QuantumCircuit(self.n_qubits)
        # Feature map (Z‑feature)
        feature_map = qiskit.circuit.library.ZFeatureMap(self.n_qubits)
        qc.append(feature_map, range(self.n_qubits))

        # Layer 1: conv + pool
        conv1 = self._conv_layer(ParameterVector("c1", 3))
        pool1 = self._pool_layer(ParameterVector("p1", 3))
        qc.compose(conv1, qubits=[0, 1], inplace=True)
        qc.compose(pool1, qubits=[2, 3], inplace=True)

        # Layer 2: conv + pool
        conv2 = self._conv_layer(ParameterVector("c2", 3))
        pool2 = self._pool_layer(ParameterVector("p2", 3))
        qc.compose(conv2, qubits=[0, 1], inplace=True)
        qc.compose(pool2, qubits=[2, 3], inplace=True)

        # Final measurement
        qc.measure_all()
        return qc

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> np.ndarray:
        """
        Execute the QCNN ansatz and return a probability distribution.

        Parameters
        ----------
        rotation_params : array of shape (12,)
            Parameters for the convolutional rotations.
        entangle_params : array of shape (6,)
            Parameters for the entangling gates (unused in this simplified ansatz).
        shots : int
            Number of measurement shots.

        Returns
        -------
        probs : array of shape (4,)
            Normalised attention weights over the four tokens.
        """
        # Bind parameters (flatten rotation_params into the circuit's
        # parameter vector; entangle_params are ignored for brevity)
        param_bindings = {
            f"c1_{i}": rotation_params[i] for i in range(3)
        }
        param_bindings.update({
            f"p1_{i}": rotation_params[3 + i] for i in range(3)
        })
        param_bindings.update({
            f"c2_{i}": rotation_params[6 + i] for i in range(3)
        })
        param_bindings.update({
            f"p2_{i}": rotation_params[9 + i] for i in range(3)
        })

        bound_circuit = self.circuit.bind_parameters(param_bindings)
        job = qiskit.execute(bound_circuit, self.backend, shots=shots)
        counts = job.result().get_counts(bound_circuit)
        # Convert counts to probabilities
        probs = np.array([counts.get(f"{i:04b}", 0) for i in range(2**self.n_qubits)], dtype=float)
        probs /= probs.sum()
        # Collapse to first 4 probabilities (one per token)
        return probs[:self.n_qubits]

__all__ = ["HybridSelfAttention"]
