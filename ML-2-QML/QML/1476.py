"""
Two‑qubit variational quantum regressor with entanglement layers.
Implemented with Pennylane; the circuit accepts a 2‑D input vector
and a 2×2 weight matrix.  The class is callable as a QNode.
"""

import pennylane as qml
import torch

class EstimatorQNNGen033:
    """
    QNode wrapper for a 2‑qubit VQC.
    The circuit encodes two input features with RX/RY gates
    and applies two entanglement‑parameterised layers.
    """
    def __init__(self) -> None:
        self.device = qml.device("default.qubit", wires=2)
        self.weight_shapes = {"weights": (2, 2)}

        @qml.qnode(self.device, interface="torch")
        def circuit(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
            # Data‑encoding
            qml.RX(inputs[0], wires=0)
            qml.RY(inputs[1], wires=1)

            # Parameterised entanglement layers
            for w in weights:
                qml.CNOT(0, 1)
                qml.RZ(w[0], wires=0)
                qml.RZ(w[1], wires=1)

            # Measurement
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def __call__(self, inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Evaluate the circuit."""
        return self.circuit(inputs, weights)
