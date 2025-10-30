"""Variational quantum regressor using Pennylane."""
import pennylane as qml
import numpy as np
from typing import Optional

class EstimatorQNN:
    """
    Parameterized quantum circuit for regression tasks.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the ansatz.
    layers : int
        Number of repeat layers of rotations and entanglement.
    entanglement : str
        Entanglement pattern ('circular', 'full', or 'linear').
    dev_name : str
        Pennylane device name.
    seed : Optional[int]
        Random seed for weight initialization.

    Methods
    -------
    __call__(inputs)
        Evaluate the circuit on the given inputs.
    """
    def __init__(
        self,
        num_qubits: int = 2,
        layers: int = 2,
        entanglement: str = "circular",
        dev_name: str = "default.qubit",
        seed: Optional[int] = None,
    ):
        self.num_qubits = num_qubits
        self.layers = layers
        self.entanglement = entanglement
        self.dev = qml.device(dev_name, wires=num_qubits)

        # Initialize trainable weights
        rng = np.random.default_rng(seed)
        self.weights = rng.uniform(-np.pi, np.pi, size=(layers * num_qubits,))

        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs: np.ndarray, weights: np.ndarray) -> qml.numpy.Tensor:
            # Encode inputs as rotations on each qubit
            for i, x in enumerate(inputs):
                qml.RX(x, wires=i)
            # Parameterized layers
            weight_idx = 0
            for _ in range(layers):
                # Single‑qubit rotations
                for w in range(num_qubits):
                    qml.RX(weights[weight_idx], wires=w)
                    weight_idx += 1
                # Entanglement
                if entanglement == "circular":
                    for w in range(num_qubits):
                        qml.CNOT(wires=[w, (w + 1) % num_qubits])
                elif entanglement == "full":
                    for w1 in range(num_qubits):
                        for w2 in range(w1 + 1, num_qubits):
                            qml.CNOT(wires=[w1, w2])
                elif entanglement == "linear":
                    for w in range(num_qubits - 1):
                        qml.CNOT(wires=[w, w + 1])
                else:
                    raise ValueError(f"Unsupported entanglement: {entanglement}")
            # Measurement on the last qubit
            return qml.expval(qml.PauliZ(num_qubits - 1))

        self.circuit = circuit

    def __call__(self, inputs: np.ndarray) -> qml.numpy.Tensor:
        """
        Evaluate the variational circuit on the supplied inputs.

        Parameters
        ----------
        inputs : np.ndarray
            1‑D array of input features.

        Returns
        -------
        qml.numpy.Tensor
            Expectation value of the chosen observable.
        """
        return self.circuit(inputs, self.weights)

__all__ = ["EstimatorQNN"]
