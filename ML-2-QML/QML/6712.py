"""
Quantum attention helper using Pennylane.

The routine encodes each input vector into a parameterised rotation
on a register of qubits, applies a tunable entangling layer,
and measures the expectation of Pauli‑Z on every qubit.  The resulting
expectation values are collapsed into a scalar attention weight
which is then used to construct an attention matrix via outer product.
"""

import numpy as np
import pennylane as qml

__all__ = ["quantum_attention"]

def quantum_attention(
    inputs: np.ndarray,
    rotation_params: np.ndarray,
    entangle_params: np.ndarray,
    n_qubits: int = 4,
    backend=None,
) -> np.ndarray:
    """
    Compute a quantum attention score matrix for a batch of input vectors.

    Parameters
    ----------
    inputs : np.ndarray
        Array of shape (N, embed_dim) where N = batch * seq_len.
    rotation_params : np.ndarray
        Rotation parameters for each qubit (length 3 * n_qubits).
    entangle_params : np.ndarray
        Entangling parameters (length n_qubits - 1).
    n_qubits : int, default 4
        Number of qubits used in the circuit.
    backend : str or qml.Device, optional
        Backend to run the circuit on.  If None, a default Pennylane
        device is used.

    Returns
    -------
    np.ndarray
        Attention score matrix of shape (N, N).
    """
    # Default device
    if backend is None:
        dev = qml.device("default.qubit", wires=n_qubits)
    else:
        dev = backend

    @qml.qnode(dev)
    def circuit(x):
        # Encode the input vector into rotations
        for i in range(min(n_qubits, len(x))):
            angle = x[i]
            w = i
            qml.RX(rotation_params[3 * w] * angle, wires=w)
            qml.RY(rotation_params[3 * w + 1] * angle, wires=w)
            qml.RZ(rotation_params[3 * w + 2] * angle, wires=w)
        # Entangling layer
        for i in range(n_qubits - 1):
            qml.CRX(entangle_params[i], wires=[i, i + 1])
        # Measure expectation of Pauli‑Z on each qubit
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    # Compute expectation values for each input vector
    expectations = []
    for vec in inputs:
        expectations.append(circuit(vec[:n_qubits]))
    expectations = np.array(expectations)  # (N, n_qubits)

    # Collapse to a scalar weight per input
    weights = np.sum(np.abs(expectations), axis=1)  # (N,)
    # Normalise
    if weights.sum() > 0:
        weights = weights / weights.sum()
    else:
        weights = np.ones_like(weights) / len(weights)

    # Build attention matrix as outer product
    scores = np.outer(weights, weights)  # (N, N)
    return scores
