"""Hybrid quantum self‑attention with EstimatorQNN.

The module builds a parameterised attention circuit and evaluates
a Qiskit EstimatorQNN that maps the circuit output to a scalar
prediction.  It is compatible with any qiskit backend that supports
state‑vector or qasm simulation.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator

class HybridSelfAttention:
    """
    Quantum hybrid self‑attention model.

    Parameters
    ----------
    n_qubits : int
        Number of qubits used for the attention circuit.
    backend : qiskit.providers.Backend, optional
        Qiskit backend for execution.  Defaults to Aer qasm simulator.
    """

    def __init__(self, n_qubits: int = 4, backend=None):
        self.n_qubits = n_qubits
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self._build_circuit_template()

    def _build_circuit_template(self):
        """
        Create a circuit template with symbolic rotation and entanglement
        parameters.  The template is reused for every forward pass.
        """
        self.qr = QuantumRegister(self.n_qubits, "q")
        self.cr = ClassicalRegister(self.n_qubits, "c")
        self.circuit_template = QuantumCircuit(self.qr, self.cr)

        # Rotation parameters per qubit
        self.rotation_params = [
            Parameter(f"rot_{i}_{axis}") for i in range(self.n_qubits) for axis in ("x", "y", "z")
        ]

        # Entanglement parameters between adjacent qubits
        self.entangle_params = [
            Parameter(f"ent_{i}") for i in range(self.n_qubits - 1)
        ]

        # Apply rotations
        for i, (rx, ry, rz) in enumerate(zip(
            self.rotation_params[0::3],
            self.rotation_params[1::3],
            self.rotation_params[2::3],
        )):
            self.circuit_template.rx(rx, i)
            self.circuit_template.ry(ry, i)
            self.circuit_template.rz(rz, i)

        # Entanglement via controlled‑RZ
        for i, ent in enumerate(self.entangle_params):
            self.circuit_template.crx(ent, i, i + 1)

        # Measurement (not used for expectation but kept for completeness)
        self.circuit_template.measure(self.qr, self.cr)

        # EstimatorQNN construction
        obs = qiskit.quantum_info.SparsePauliOp.from_list([("Y" * self.n_qubits, 1)])
        self.estimator = StatevectorEstimator()
        self.estimator_qnn = EstimatorQNN(
            circuit=self.circuit_template,
            observables=obs,
            input_params=self.rotation_params[0::3],  # encode inputs via X rotations
            weight_params=self.rotation_params[1::3],  # weights via Y rotations
            estimator=self.estimator,
        )

    def _parameter_bindings(self, rotation_vals, entangle_vals):
        """
        Create a dictionary mapping symbolic parameters to numeric values.
        """
        bindings = {}
        # rotation_vals: shape (n_qubits, 3)
        for i in range(self.n_qubits):
            bindings[self.rotation_params[3 * i]] = rotation_vals[i, 0]
            bindings[self.rotation_params[3 * i + 1]] = rotation_vals[i, 1]
            bindings[self.rotation_params[3 * i + 2]] = rotation_vals[i, 2]
        # entangle_vals: shape (n_qubits-1,)
        for i in range(self.n_qubits - 1):
            bindings[self.entangle_params[i]] = entangle_vals[i]
        return bindings

    def run(
        self,
        rotation_vals: np.ndarray,
        entangle_vals: np.ndarray,
        input_vector: np.ndarray,
        shots: int = 1024,
    ) -> float:
        """
        Execute the attention circuit and evaluate the EstimatorQNN.

        Parameters
        ----------
        rotation_vals : np.ndarray
            Rotation angles of shape (n_qubits, 3).
        entangle_vals : np.ndarray
            Entanglement angles of shape (n_qubits-1,).
        input_vector : np.ndarray
            Classical input to be encoded via X rotations.
        shots : int
            Number of shots for the qasm simulator.

        Returns
        -------
        float
            Expectation value output by the EstimatorQNN.
        """
        # Encode classical input into rotation angles (simple mapping)
        # For demonstration, map input_vector to first axis rotations.
        rotation_vals[:, 0] = input_vector[: self.n_qubits]

        bindings = self._parameter_bindings(rotation_vals, entangle_vals)

        # Bind parameters and execute
        bound_circuit = self.circuit_template.bind_parameters(bindings)
        job = qiskit.execute(bound_circuit, self.backend, shots=shots)
        counts = job.result().get_counts(bound_circuit)

        # Convert counts to expectation value via estimator
        expectation = self.estimator_qnn.predict(counts)[0]
        return float(expectation)

def get_hybrid_self_attention(n_qubits: int = 4, backend=None):
    """
    Factory function returning a ready‑to‑use HybridSelfAttention instance.
    """
    return HybridSelfAttention(n_qubits=n_qubits, backend=backend)
