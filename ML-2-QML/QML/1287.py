import pennylane as qml
import numpy as np

class SelfAttention:
    """
    Quantum self‑attention implemented with PennyLane.
    The circuit returns a probability distribution over the wires, which can be used
    as attention weights.  The interface mirrors the classical version:
    run(backend, rotation_params, entangle_params, shots).
    """

    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim
        # Default device; can be overridden in run
        self.device = qml.device("default.qubit", wires=embed_dim)

    def run(self, backend, rotation_params: np.ndarray, entangle_params: np.ndarray, shots: int = 1024):
        """
        Parameters
        ----------
        backend : qml.Device
            PennyLane device or simulator.
        rotation_params : np.ndarray
            Rotation angles for Ry on each wire. Shape (embed_dim,).
        entangle_params : np.ndarray
            Rotation angles for Rz used as a simple entanglement modulator.
            Shape (embed_dim,).
        shots : int
            Number of shots for sampling (ignored for state‑vector devices).

        Returns
        -------
        np.ndarray
            Attention‑like probability distribution over the wires.
        """
        self.device = backend

        @qml.qnode(self.device, interface="numpy")
        def circuit():
            # Apply parameterised rotations
            for i in range(self.embed_dim):
                qml.RY(rotation_params[i], wires=i)
                qml.RZ(entangle_params[i], wires=i)
            # Simple linear entanglement
            for i in range(self.embed_dim - 1):
                qml.CNOT(wires=[i, i + 1])
            # Measure expectation of PauliZ on each wire
            return [qml.expval(qml.PauliZ(i)) for i in range(self.embed_dim)]

        raw_expectations = circuit()
        # Convert to a probability distribution
        probs = np.exp(raw_expectations) / np.sum(np.exp(raw_expectations))
        return probs

__all__ = ["SelfAttention"]
