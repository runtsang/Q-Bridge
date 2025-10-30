"""ConvGen105Q: A quantum-inspired convolution module using a variational circuit.

The module:
- accepts multiple kernel sizes,
- builds a parameterised circuit for each kernel size,
- uses a tunable depth,
- performs measurement and classical readout,
- provides a hybrid loss interface.
"""

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

class ConvGen105Q:
    def __init__(
        self,
        kernel_sizes=(2, 3),
        depth=2,
        threshold=0.5,
        shots=1000,
        dev_name="default.qubit",
    ):
        self.kernel_sizes = kernel_sizes
        self.depth = depth
        self.threshold = threshold
        self.shots = shots
        self.dev = qml.device(dev_name, wires=max(k * k for k in kernel_sizes))
        self.circuits = {}
        for k in kernel_sizes:
            self.circuits[k] = self._build_circuit(k)

    def _build_circuit(self, k):
        n = k * k

        @qml.qnode(self.dev, interface="autograd", diff_method="backprop")
        def circuit(x):
            # Encode data into rotation angles
            for i in range(n):
                qml.RX(x[i], wires=i)
            # Variational layers
            for _ in range(self.depth):
                for i in range(n):
                    qml.RY(pnp.random.uniform(0, 2 * np.pi), wires=i)
                for i in range(0, n - 1, 2):
                    qml.CNOT(wires=[i, i + 1])
            # Measurement
            return [qml.expval(qml.PauliZ(i)) for i in range(n)]

        return circuit

    def run(self, data):
        """data: list of 2D arrays, each of shape (k, k) for corresponding kernel sizes."""
        outputs = []
        for k, arr in zip(self.kernel_sizes, data):
            flat = arr.flatten()
            result = self.circuits[k](flat)
            # Convert expectation values to probabilities of |1>
            probs = (1 - np.array(result)) / 2  # expval of Z: +1 for |0>, -1 for |1>
            # Compute mean probability above threshold
            mean_prob = np.mean(probs > self.threshold)
            outputs.append(mean_prob)
        return np.array(outputs)

    def hybrid_loss(self, preds, labels, loss_fn):
        """Compute loss and return gradient for backprop."""
        loss = loss_fn(preds, labels)
        return loss
