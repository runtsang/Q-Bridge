"""ConvEnhanced: quantum variational filter with data‑encoded rotations and Pauli‑Z measurement."""

import pennylane as qml
import numpy as np
from pennylane import numpy as pnp


class ConvEnhanced:
    """Quantum replacement for the Conv filter.

    The circuit encodes the pixel values as rotation angles (RY), applies a stack of
    variational layers (RX, RY, RZ) with entanglement, and measures the expectation
    of Pauli‑Z on each qubit. The mean of the derived |1⟩ probabilities is returned
    as a scalar feature. The module is fully differentiable via PennyLane’s autograd
    interface, enabling gradient‑based training alongside classical layers.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        layers: int = 2,
        shots: int = 1024,
        threshold: float = 0.5,
        device_name: str = "default.qubit",
    ):
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.shots = shots
        self.threshold = threshold
        self.dev = qml.device(device_name, wires=self.n_qubits, shots=shots)
        self.layers = layers

        # Trainable parameters for each variational layer
        self.params = np.random.randn(layers, self.n_qubits, 3)  # RX, RY, RZ

        @qml.qnode(self.dev, interface="autograd")
        def circuit(inputs, params):
            # Encode input data as RY rotations
            for i, val in enumerate(inputs):
                qml.RY(val, wires=i)

            # Variational layers
            for layer in range(layers):
                for q in range(self.n_qubits):
                    qml.RX(params[layer, q, 0], wires=q)
                    qml.RY(params[layer, q, 1], wires=q)
                    qml.RZ(params[layer, q, 2], wires=q)
                # Entangle neighbours in a ring
                for q in range(self.n_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
                qml.CNOT(wires=[self.n_qubits - 1, 0])

            # Measure expectation of Pauli‑Z on all wires
            return [qml.expval(qml.PauliZ(w)) for w in range(self.n_qubits)]

        self.circuit = circuit

    def run(self, data) -> float:
        """Return the mean |1⟩ probability for the given 2‑D input."""
        flat = data.reshape(-1).astype(np.float32)
        # Normalize to [0, π] for rotation angles
        flat = np.clip(flat, 0, 255) / 255.0 * np.pi
        expvals = self.circuit(flat, self.params)
        probs = (1 - np.array(expvals)) / 2
        return float(probs.mean())

    def __call__(self, data):
        return self.run(data)


def Conv(*args, **kwargs):
    """Factory that returns a ConvEnhanced instance, keeping the original API."""
    return ConvEnhanced(*args, **kwargs)


__all__ = ["ConvEnhanced", "Conv"]
