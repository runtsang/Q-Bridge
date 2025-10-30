"""Quantum convolutional filter using a parameter‑shared variational ansatz.

The circuit encodes pixel values into rotations, applies a depth‑wise
entangling layer, and measures the expectation of Pauli‑Z on each qubit.
The average probability of measuring |1> is returned, mimicking the
classical sigmoid activation.  The circuit is differentiable with
respect to the variational parameters, enabling joint training with a
classical backbone.
"""

import pennylane as qml
import numpy as np


def ConvGen130(kernel_size: int = 2, device: str = "default.qubit", shots: int = 100, threshold: float = 0.0):
    """Return a quantum convolutional filter.

    Parameters
    ----------
    kernel_size : int
        Size of the square kernel (number of qubits = kernel_size**2).
    device : str
        PennyLane device identifier.
    shots : int
        Number of shots for sampling.
    threshold : float
        Pixel threshold to decide the encoding rotation (pi or 0).
    """

    class ConvGen130:
        def __init__(self):
            self.kernel_size = kernel_size
            self.n_qubits = kernel_size ** 2
            self.threshold = threshold
            self.device = qml.device(device, wires=self.n_qubits, shots=shots)

            # Trainable variational parameters (one per qubit)
            self.params = np.random.uniform(0, 2 * np.pi, size=self.n_qubits)

            # QNode
            @qml.qnode(self.device, interface="numpy")
            def circuit(inputs, params):
                # Encode pixel values
                for i, val in enumerate(inputs):
                    angle = np.pi if val > self.threshold else 0.0
                    qml.RY(angle, wires=i)

                # Variational ansatz
                for i in range(self.n_qubits):
                    qml.RY(params[i], wires=i)
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])

                # Measure expectation of Pauli‑Z
                return [qml.expval(qml.PauliZ(w)) for w in range(self.n_qubits)]

            self.circuit = circuit

        def run(self, data):
            """Run the quantum filter on a single kernel image.

            Parameters
            ----------
            data : 2D array of shape (kernel_size, kernel_size)

            Returns
            -------
            float
                Average probability of measuring |1> across all qubits.
            """
            data_flat = data.reshape(-1)
            expect = self.circuit(data_flat, self.params)
            probs = (1 - np.array(expect)) / 2
            return probs.mean()

    return ConvGen130()
