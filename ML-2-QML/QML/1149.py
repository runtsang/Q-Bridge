import pennylane as qml
import numpy as np
from pennylane import numpy as pnp

class SelfAttentionModule:
    """
    Variational quantum self‑attention circuit.
    The circuit encodes the input sequence into a register of qubits,
    applies rotation and entanglement layers parameterised by the
    supplied arrays, and returns the measurement probability
    distribution over the computational basis.
    """
    def __init__(self, n_qubits: int = 4):
        self.n_qubits = n_qubits
        self.dev = qml.device("default.qubit", wires=n_qubits)

    def _encode(self, inputs: np.ndarray):
        """Angle‑embedding of the input features."""
        for i, x in enumerate(inputs):
            qml.RX(x, wires=i)
            qml.RY(x, wires=i)
            qml.RZ(x, wires=i)

    def _rotation_layer(self, params: np.ndarray):
        """Apply a layer of single‑qubit rotations."""
        for i, (rx, ry, rz) in enumerate(params):
            qml.RX(rx, wires=i)
            qml.RY(ry, wires=i)
            qml.RZ(rz, wires=i)

    def _entanglement_layer(self, params: np.ndarray):
        """Entangle neighbouring qubits with controlled‑RZ gates."""
        for i, phi in enumerate(params):
            qml.CRX(phi, wires=[i, i + 1])

    def _circuit(self, rotation_params, entangle_params, inputs):
        self._encode(inputs)
        self._rotation_layer(rotation_params)
        self._entanglement_layer(entangle_params)

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
        shots: int = 1024,
    ) -> np.ndarray:
        """
        Execute the circuit and return the probability distribution
        over all basis states.  The returned array has shape
        ``(2**n_qubits,)`` and is sorted lexicographically.
        """
        @qml.qnode(self.dev, interface="autograd", shots=shots)
        def circuit():
            self._circuit(rotation_params, entangle_params, inputs)
            return qml.probs(wires=range(self.n_qubits))

        probs = circuit()
        return probs.detach().numpy()

__all__ = ["SelfAttentionModule"]
