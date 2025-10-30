"""Quantum variational filter that learns a 2×2 kernel via a parameter‑efficient circuit."""

import numpy as np
import pennylane as qml
from pennylane import numpy as qnp

def ConvGen050():
    """Return a callable class that implements a variational quanvolution.

    The circuit is a 2×2 grid of qubits.  Each qubit receives a
    rotation around the Y‑axis parameterised by a trainable weight
    ``theta``.  The circuit then applies a fixed two‑layer
    entangling‑gate structure (CX‑CNOT) that couples all qubits.  The
    expectation value of the Pauli‑Z measurement on each qubit is
    returned as a 2‑D array that can be used as a convolutional
    kernel.  The whole circuit can be device‑agnostic and
    (in‑place)‐re‑parameterised to support gradient‑based
    optimisation.
    """

    class ConvFilter:
        """Quantum‑classical hybrid filter with one trainable parameter per
        2×2 patch.  The target metric is the average of Z‑expectations across
        all qubits, optionally thresholded by a classical value.
        """

        def __init__(self, kernel_size=2, threshold=0.0, device=None):
            self.kernel_size = kernel_size
            self.threshold = threshold
            self.n_qubits = kernel_size ** 2
            self.dev = device or qml.device("default.qubit", wires=self.n_qubits)
            # Trainable parameters
            self.params = qnp.random.randn(self.n_qubits)

            @qml.qnode(self.dev, interface="autograd")
            def circuit(theta, data):
                # Encode data as Y‑rotations
                for i in range(self.n_qubits):
                    qml.RY(theta[i] * data[i], wires=i)
                # Entangling layer
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                # Second entangling layer
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i + 1, i])
                # Measure Z on all qubits
                return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

            self.circuit = circuit

        def run(self, data):
            """Run the quantum circuit on a 2×2 patch.

            Args:
                data (np.ndarray): 2D array with shape (kernel_size, kernel_size).

            Returns:
                float: average expectation value of Z across qubits, optionally
                thresholded.
            """
            flat = data.flatten()
            expvals = self.circuit(self.params, flat)
            avg = np.mean(expvals)
            return float(avg) if avg > self.threshold else 0.0

        def train_step(self, data, lr=0.01):
            """Perform a single gradient‑descent step on the parameters."""
            flat = data.flatten()

            def loss_fn(theta):
                expvals = self.circuit(theta, flat)
                return -np.mean(expvals)  # maximise average Z

            grads = qml.grad(loss_fn)(self.params)
            self.params -= lr * grads

    return ConvFilter()
