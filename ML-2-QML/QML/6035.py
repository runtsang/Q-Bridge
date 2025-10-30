"""
Quantum phase generator for hybrid self‑attention.
"""

import pennylane as qml
import numpy as np

def quantum_phase_vector(
    rotation_params: np.ndarray,
    entangle_params: np.ndarray,
    inputs: np.ndarray,
) -> np.ndarray:
    """
    Return a phase vector of shape (embed_dim,) produced by a variational circuit.

    Parameters
    ----------
    rotation_params : np.ndarray
        Rotation parameters for RX, RY, RZ gates. Shape (3*embed_dim,).
    entangle_params : np.ndarray
        Parameters for CRX entangling gates. Shape (embed_dim-1,).
    inputs : np.ndarray
        Input vector of shape (embed_dim,). Each element is encoded as a Z‑rotation.

    Returns
    -------
    np.ndarray
        Phase vector of shape (embed_dim,).
    """
    embed_dim = inputs.shape[0]
    dev = qml.device("default.qubit", wires=embed_dim)

    @qml.qnode(dev, interface="autograd")
    def circuit():
        # Encode inputs as Z rotations
        for i in range(embed_dim):
            qml.RZ(inputs[i], wires=i)
        # Parameterized single‑qubit rotations
        for i in range(embed_dim):
            qml.RX(rotation_params[3 * i], wires=i)
            qml.RY(rotation_params[3 * i + 1], wires=i)
            qml.RZ(rotation_params[3 * i + 2], wires=i)
        # Entangling layer
        for i in range(embed_dim - 1):
            qml.CRX(entangle_params[i], wires=[i, i + 1])
        # Return Z expectation values as phase vector
        return [qml.expval(qml.PauliZ(i)) for i in range(embed_dim)]

    return np.array(circuit())

class QuantumSelfAttention:
    """
    Wrapper that exposes the quantum phase vector as a callable compatible with SelfAttentionHybrid.
    """

    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        return quantum_phase_vector(rotation_params, entangle_params, inputs)

__all__ = ["quantum_phase_vector", "QuantumSelfAttention"]
