import pennylane as qml
import torch
from torch import nn
import pennylane.numpy as np

class EstimatorQNN(nn.Module):
    """
    Hybrid quantum‑classical neural network for regression.

    The circuit encodes a 4‑qubit input into a variational ansatz
    using strongly entangling layers.  The expectation value of
    Pauli‑Z on the last qubit is returned as the regression output.
    The model is fully differentiable via Torch and can be trained
    end‑to‑end with any standard loss function.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        device: str = "default.qubit",
    ) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.device = device

        # PennyLane quantum device
        self.dev = qml.device(device, wires=n_qubits)

        # QNode with Torch autograd
        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
            # Feature encoding: rotate each qubit by input value
            for i in range(n_qubits):
                qml.RX(np.pi * inputs[i], wires=i)

            # Variational ansatz
            qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))

            # Measurement
            return qml.expval(qml.PauliZ(n_qubits - 1))

        self.circuit = circuit

        # Initialise trainable weights
        weight_shape = (n_layers, n_qubits, 3)
        self.weights = nn.Parameter(0.01 * torch.randn(weight_shape))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the hybrid model.

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape (batch, n_qubits).  Input features should be
            normalised to the interval [0, 1].

        Returns
        -------
        torch.Tensor
            Tensor of shape (batch, 1) containing the regression output.
        """
        batch_size = inputs.shape[0]
        outputs = []
        for i in range(batch_size):
            out = self.circuit(inputs[i], self.weights)
            outputs.append(out)
        return torch.stack(outputs).unsqueeze(-1)

__all__ = ["EstimatorQNN"]
