"""
Quantum feature extractor for the hybrid estimator.
Implements a 4‑qubit circuit with input‑dependent Ry rotations
followed by a fixed entangling layer.  The circuit is evaluated
on the state‑vector simulator and the expectation values of
Pauli‑Z on each qubit are returned as the feature vector.
"""

from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import Statevector, Pauli
import numpy as np
import torch

def quantum_kernel(inputs: torch.Tensor) -> torch.Tensor:
    """
    Compute quantum kernel features for a batch of 4‑dimensional inputs.

    Parameters
    ----------
    inputs : torch.Tensor
        Tensor of shape (batch, 4) containing real‑valued input features.

    Returns
    -------
    torch.Tensor
        Tensor of shape (batch, 4) with expectation values of Pauli‑Z
        on each qubit of the quantum circuit.
    """
    batch, n_qubits = inputs.shape
    backend = Aer.get_backend('statevector_simulator')
    feature_vectors = []

    for i in range(batch):
        # Convert input to NumPy array (CPU)
        vec = inputs[i].detach().cpu().numpy()
        qc = QuantumCircuit(n_qubits)

        # Encode the input features with Ry rotations
        for q in range(n_qubits):
            qc.ry(vec[q], q)

        # Fixed entangling layer (CNOT chain + wrap‑around)
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)
        qc.cx(n_qubits - 1, 0)

        # Simulate the circuit to obtain the statevector
        result = execute(qc, backend).result()
        sv = Statevector(result.get_statevector(qc))

        # Compute expectation values of Pauli‑Z on each qubit
        exp_vals = []
        for q in range(n_qubits):
            pauli_str = 'I' * q + 'Z' + 'I' * (n_qubits - q - 1)
            pauli = Pauli(pauli_str)
            exp = sv.expectation_value(pauli).real
            exp_vals.append(exp)

        feature_vectors.append(exp_vals)

    return torch.tensor(feature_vectors, dtype=torch.float32)

__all__ = ["quantum_kernel"]
