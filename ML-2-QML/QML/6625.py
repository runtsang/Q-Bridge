"""Quantum self‑attention implementation using Qiskit.

The module provides a variational circuit that can be used to compute
attention weights. It exposes two public functions:

- get_quantum_circuit: builds a parameterised circuit.
- run_quantum_attention: runs the circuit on a backend and returns
  statevector amplitudes or measurement counts.

The design is intentionally lightweight and can be swapped out for
other backends (PennyLane, Braket) with minimal changes.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.providers.aer import AerSimulator

def get_quantum_circuit(embed_dim: int, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
    """
    Build a variational circuit representing a self‑attention block.

    Parameters
    ----------
    embed_dim : int
        Number of qubits / embedding dimension.
    rotation_params : np.ndarray
        Array of shape (embed_dim*3,) containing rx, ry, rz angles.
    entangle_params : np.ndarray
        Array of shape (embed_dim-1,) containing controlled‑rx angles.

    Returns
    -------
    QuantumCircuit
        The constructed circuit.
    """
    qr = QuantumRegister(embed_dim, "q")
    cr = ClassicalRegister(embed_dim, "c")
    circuit = QuantumCircuit(qr, cr)

    # Apply single‑qubit rotations
    for i in range(embed_dim):
        circuit.rx(rotation_params[3 * i], qr[i])
        circuit.ry(rotation_params[3 * i + 1], qr[i])
        circuit.rz(rotation_params[3 * i + 2], qr[i])

    # Entangling layer
    for i in range(embed_dim - 1):
        circuit.crx(entangle_params[i], qr[i], qr[i + 1])

    return circuit

def run_quantum_attention(inputs: np.ndarray,
                          rotation_params: np.ndarray,
                          entangle_params: np.ndarray,
                          backend: qiskit.providers.Backend | None = None,
                          shots: int = 1024) -> np.ndarray:
    """
    Execute the attention circuit for a batch of inputs.

    Parameters
    ----------
    inputs : np.ndarray
        Shape (batch, embed_dim). Each row is treated as additional
        rotation angles added to rotation_params.
    rotation_params : np.ndarray
        Base rotation angles for the circuit.
    entangle_params : np.ndarray
        Entangling angles.
    backend : qiskit.providers.Backend, optional
        Backend to execute on. Defaults to Aer statevector simulator.
    shots : int
        Number of shots for measurement.

    Returns
    -------
    np.ndarray
        For statevector simulation: array of shape (batch, 2**embed_dim).
        For measurement simulation: array of dicts with counts per batch.
    """
    if backend is None:
        backend = AerSimulator(method="statevector")

    batch_size, embed_dim = inputs.shape
    results = []

    for i in range(batch_size):
        # Combine base rotation with input encoding
        rot = rotation_params + inputs[i]
        circuit = get_quantum_circuit(embed_dim, rot, entangle_params)

        if isinstance(backend, AerSimulator):
            result = backend.run(circuit).result()
            statevec = result.get_statevector(circuit)
            results.append(statevec)
        else:
            job = qiskit.execute(circuit, backend, shots=shots)
            counts = job.result().get_counts(circuit)
            results.append(counts)

    return np.array(results)

__all__ = ["get_quantum_circuit", "run_quantum_attention"]
