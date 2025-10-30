import pennylane as qml
import torch
from torch import nn
from typing import Tuple

class QuantumDecoder(nn.Module):
    """
    Quantum decoder implemented as a PennyLane QNode.

    Parameters
    ----------
    num_qubits : int
        Number of qubits used for the decoder. One qubit per output dimension.
    latent_dim : int
        Dimension of the latent vector coming from the classical encoder.
    n_layers : int, default=1
        Number of variational layers in the ansatz.
    seed : int, default=42
        Random seed for device initialization.
    """

    def __init__(self,
                 num_qubits: int,
                 latent_dim: int,
                 n_layers: int = 1,
                 seed: int = 42) -> None:
        super().__init__()
        self.num_qubits = num_qubits
        self.latent_dim = latent_dim
        self.n_layers = n_layers

        # Device
        self.dev = qml.device("default.qubit", wires=self.num_qubits, shots=None, seed=seed)

        # Trainable parameters for the ansatz
        self.weight_params = nn.Parameter(
            torch.randn(self.n_layers, self.num_qubits, 3)
        )

        # QNode definition
        self.qnode = qml.QNode(self._circuit,
                               self.dev,
                               interface="torch")

    def _circuit(self, latent: torch.Tensor,
                 weight_params: torch.Tensor) -> torch.Tensor:
        """
        Circuit that encodes the latent vector and applies a variational ansatz.

        Parameters
        ----------
        latent : torch.Tensor
            Latent vector of shape (latent_dim,).
        weight_params : torch.Tensor
            Trainable parameters of shape (n_layers, num_qubits, 3).
        """
        # Encode latent into qubits
        for i in range(min(self.num_qubits, self.latent_dim)):
            qml.RY(latent[i], wires=i)

        # Variational layers
        for layer in range(self.n_layers):
            for q in range(self.num_qubits):
                qml.RZ(weight_params[layer, q, 0], wires=q)
                qml.RX(weight_params[layer, q, 1], wires=q)
                qml.RY(weight_params[layer, q, 2], wires=q)
            # Entangling layer
            for q in range(self.num_qubits - 1):
                qml.CNOT(wires=[q, q + 1])

        # Return probability of measuring |0> for each qubit
        probs = torch.stack([qml.probs(wires=q)[0] for q in range(self.num_qubits)])
        return probs

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the quantum decoder.

        Parameters
        ----------
        latent : torch.Tensor
            Latent vector of shape (latent_dim,).
        """
        return self.qnode(latent, self.weight_params)
