"""Hybrid quantum self‑attention and classifier implementation.

The circuit combines the self‑attention style ansatz from SelfAttention.py
with the incremental data‑uploading classifier from
QuantumClassifierModel.py.  The encoding gates are the same rx gates used
in the classifier; the self‑attention rotations (rx, ry, rz) and CRX
entanglement are inserted before the variational layers (ry, cz).
Observables are Pauli‑Z on each qubit, allowing a measurement of the
class‑dependent expectation values.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from typing import Iterable

def build_classifier_circuit(num_qubits: int, depth: int) -> tuple[QuantumCircuit, Iterable, Iterable, list[SparsePauliOp]]:
    """
    Build a hybrid self‑attention + classifier quantum circuit.

    Parameters
    ----------
    num_qubits : int
        Number of qubits (feature dimension).
    depth : int
        Number of variational layers after the self‑attention block.
    """
    # Data encoding
    encoding = ParameterVector("x", num_qubits)

    # Self‑attention parameters
    rotation = ParameterVector("theta_a", 3 * num_qubits)
    entangle = ParameterVector("phi_a", num_qubits - 1)

    # Variational parameters
    var_weights = ParameterVector("theta_v", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)

    # --- Encoding ---
    for i, param in enumerate(encoding):
        circuit.rx(param, i)

    # --- Self‑attention block ---
    for i in range(num_qubits):
        circuit.rx(rotation[3 * i], i)
        circuit.ry(rotation[3 * i + 1], i)
        circuit.rz(rotation[3 * i + 2], i)

    for i in range(num_qubits - 1):
        circuit.crx(entangle[i], i, i + 1)

    # --- Variational layers (classifier) ---
    idx = 0
    for _ in range(depth):
        for i in range(num_qubits):
            circuit.ry(var_weights[idx], i)
            idx += 1
        for i in range(num_qubits - 1):
            circuit.cz(i, i + 1)

    # --- Observables ---
    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]

    return circuit, list(encoding), list(rotation) + list(entangle) + list(var_weights), observables

class QuantumHybridSelfAttention:
    """
    Wrapper around the hybrid self‑attention + classifier circuit.
    """

    def __init__(self, num_qubits: int, depth: int, backend=None, shots: int = 1024):
        self.num_qubits = num_qubits
        self.depth = depth
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.circuit, self.encoding, self.parameters, self.observables = build_classifier_circuit(num_qubits, depth)

    def run(self, data: np.ndarray, params: np.ndarray) -> dict:
        """
        Execute the circuit for a single data point.

        Parameters
        ----------
        data : np.ndarray
            Array of shape (num_qubits,) containing the feature values.
        params : np.ndarray
            Array of variational parameters matching the circuit.
        """
        param_dict = {p: v for p, v in zip(self.encoding, data)}
        param_dict.update({p: v for p, v in zip(self.parameters, params)})
        bound_circuit = self.circuit.bind_parameters(param_dict)
        job = qiskit.execute(bound_circuit, self.backend, shots=self.shots)
        return job.result().get_counts(bound_circuit)

__all__ = ["QuantumHybridSelfAttention", "build_classifier_circuit"]
