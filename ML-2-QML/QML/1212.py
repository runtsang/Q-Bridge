"""
EstimatorQNN__gen153 – Quantum variational regressor built with Pennylane.
"""

import pennylane as qml
from pennylane import numpy as np
from typing import Sequence, Tuple, Callable


class EstimatorQNN__gen153:
    """
    Variational quantum circuit that maps two classical inputs to a single
    expectation value of a Pauli‑Z observable.

    The circuit uses a 2‑qubit ansatz with alternating RX/RZ layers and a
    trainable rotation on each qubit.  Inputs are encoded via
    parameterised RX gates.  The circuit is differentiable via the
    parameter‑shift rule and can be used in hybrid training loops.
    """

    def __init__(
        self,
        hidden_layers: Sequence[int] = (2, 2),
        device_name: str = "default.qubit",
        shots: int = 1024,
    ) -> None:
        self.num_qubits = 2
        self.device = qml.device(device_name, wires=self.num_qubits, shots=shots)

        # Trainable parameters: one per qubit per hidden layer
        self.params = qml.numpy.array(
            [np.random.uniform(0, 2 * np.pi, size=h) for h in hidden_layers],
            dtype=np.float64,
        )

        # Build the QNode
        @qml.qnode(self.device, interface="autograd", diff_method="parameter-shift")
        def circuit(inputs: Tuple[float, float], params: np.ndarray) -> float:
            # Input encoding
            qml.RX(inputs[0], wires=0)
            qml.RX(inputs[1], wires=1)

            # Ansatz
            for layer_idx, layer_params in enumerate(params):
                for qubit, theta in enumerate(layer_params):
                    qml.RZ(theta, wires=qubit)
                # Entangling layer
                qml.CNOT(wires=[0, 1])

            # Measurement
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Parameters
        ----------
        inputs : np.ndarray
            Array of shape ``(batch_size, 2)`` containing two real features.

        Returns
        -------
        np.ndarray
            Array of shape ``(batch_size,)`` with the predicted outputs.
        """
        outputs = []
        for inp in inputs:
            outputs.append(self.circuit(inp, self.params))
        return np.array(outputs)

    def train_step(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        lr: float = 0.01,
    ) -> float:
        """
        One gradient‑descent step using the MSE loss.

        Parameters
        ----------
        inputs : np.ndarray
            Batch of input features.
        targets : np.ndarray
            Ground‑truth regression targets.
        lr : float
            Learning rate.

        Returns
        -------
        float
            Current loss value.
        """
        loss = np.mean((self(inputs) - targets) ** 2)

        # Compute gradients w.r.t. trainable parameters
        grads = qml.grad(self.circuit, argnum=1)(inputs, self.params)
        # Update parameters
        self.params -= lr * grads
        return float(loss)


def EstimatorQNN__gen153() -> EstimatorQNN__gen153:
    """
    Factory that returns a ready‑to‑train instance with default hyper‑parameters.
    """
    return EstimatorQNN__gen153()


__all__ = ["EstimatorQNN__gen153"]
