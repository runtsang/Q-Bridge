"""Quantum convolutional filter using a variational circuit.

The class QuanvCircuit implements a 2‑D filter that encodes the input data into
rotation angles and applies a small variational ansatz.  The output is the
expectation value of a PauliZ measurement on the first qubit, which can be
interpreted as a probability after a sigmoid transform.

Typical usage::

    from Conv__gen154 import Conv
    qc = Conv()
    out = qc.run(data)

"""

import pennylane as qml
import numpy as np
from pennylane import numpy as pnp

class QuanvCircuit:
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0, num_layers: int = 2, shots: int = 100):
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.num_qubits = kernel_size ** 2
        self.device = qml.device("default.qubit", wires=self.num_qubits, shots=shots)
        self.num_layers = num_layers
        # initialise parameters
        self.params = pnp.random.uniform(0, 2 * pnp.pi, size=(num_layers, self.num_qubits))

        @qml.qnode(self.device, interface="torch")
        def circuit(params, x):
            # Encode input
            for i in range(self.num_qubits):
                qml.RY(x[i], wires=i)
            # Variational layers
            for layer in range(params.shape[0]):
                for i in range(params.shape[1]):
                    qml.RZ(params[layer, i], wires=i)
                # Entangling CNOT chain
                for i in range(self.num_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            # Return expectation of PauliZ on first qubit
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def run(self, data):
        """Run the quantum filter on a 2‑D array.

        Parameters
        ----------
        data : array‑like
            Input of shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Filter response.
        """
        x = np.array(data).reshape(-1)
        # Encode threshold: values > threshold -> π, else 0
        x = np.where(x > self.threshold, np.pi, 0.0)
        # Convert to torch tensor
        import torch
        x_torch = torch.tensor(x, dtype=torch.float32)
        out = self.circuit(self.params, x_torch)
        return out.item()

def Conv():
    """Return a QuanvCircuit instance with default parameters."""
    return QuanvCircuit()
