"""Quantum Conv: parameterised ansatz for quanvolution with adaptive measurement."""
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


class ConvQuantum:
    """
    Parameterised quantum circuit that mimics a convolutional filter.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the square kernel (number of qubits = kernel_size**2).
    num_layers : int, default 2
        Number of ansatz layers.
    shots : int, default 1024
        Number of shots for simulation.
    backend : str, default "default.qubit"
        PennyLane device name.
    threshold : float, default 0.5
        Threshold for converting expectation to binary output.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        *,
        num_layers: int = 2,
        shots: int = 1024,
        backend: str = "default.qubit",
        threshold: float = 0.5,
    ) -> None:
        self.kernel_size = kernel_size
        self.num_qubits = kernel_size ** 2
        self.num_layers = num_layers
        self.shots = shots
        self.threshold = threshold

        self.dev = qml.device(backend, wires=self.num_qubits, shots=self.shots)

        # Trainable parameters
        self.params = pnp.random.uniform(0, 2 * np.pi, size=(self.num_layers, self.num_qubits))

        @qml.qnode(self.dev, interface="autograd")
        def circuit(x, params):
            # Encode data as rotation angles
            for i, val in enumerate(x):
                qml.RX(val * np.pi, wires=i)

            # Ansatz layers
            for layer in range(self.num_layers):
                for qubit in range(self.num_qubits):
                    qml.RY(params[layer, qubit], wires=qubit)
                # Entanglement (nearest‑neighbour chain)
                for qubit in range(self.num_qubits - 1):
                    qml.CNOT(wires=[qubit, qubit + 1])

            # Measurement: expectation of PauliZ on all qubits
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

        self.circuit = circuit

    def run(self, data):
        """
        Evaluate the quantum filter on classical data.

        Parameters
        ----------
        data : array‑like
            2‑D array of shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Average probability of measuring |1> across qubits after thresholding.
        """
        x = np.reshape(data, self.num_qubits)
        expvals = self.circuit(x, self.params)
        # Convert expectation values to probabilities of |1>
        probs = (1 - np.array(expvals)) / 2
        # Apply threshold to obtain binary outputs
        binary = (probs > self.threshold).astype(float)
        return binary.mean()

    def loss(self, data, target):
        """
        Simple mean‑squared‑error loss for the quantum filter.

        Parameters
        ----------
        data : array‑like
            Input data.
        target : float
            Desired scalar output.

        Returns
        -------
        float
            MSE loss.
        """
        pred = self.run(data)
        return (pred - target) ** 2

    def train_step(self, data, target, lr=0.01):
        """
        One optimisation step using the parameter‑shift rule.

        Parameters
        ----------
        data : array‑like
            Input data.
        target : float
            Desired scalar output.
        lr : float, default 0.01
            Learning rate.

        Returns
        -------
        float
            Updated loss value.
        """
        # Compute loss and gradient
        loss_val = self.loss(data, target)
        grads = qml.grad(self.loss)(data, target)
        # Update parameters
        self.params -= lr * grads
        return loss_val


__all__ = ["ConvQuantum"]
