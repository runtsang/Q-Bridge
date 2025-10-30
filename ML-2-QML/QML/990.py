import pennylane as qml
import torch
from torch import nn

class QuantumDecoder(nn.Module):
    """
    Quantum decoder that maps a latent vector to an output vector via a variational circuit.
    The circuit uses a RealAmplitudes ansatz followed by measurement of PauliZ on each output wire.
    Parameters are torch tensors and can be optimized with a classical optimizer.
    """
    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        layers: int = 2,
        qreg_size: int = 4,
        device: str = "default.qubit",
        wires: int | list[int] | None = None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.layers = layers
        self.qreg_size = qreg_size
        self.device = device

        if wires is None:
            wires = list(range(output_dim))
        self.wires = wires

        # Create a trainable weight matrix for the ansatz
        # The RealAmplitudes ansatz expects a weight vector of shape (layers, wires, 3)
        self.weight_shape = (layers, len(self.wires), 3)
        self.weights = nn.Parameter(
            torch.randn(self.weight_shape, dtype=torch.float64)
        )

        # Define a QNode that accepts a latent vector and returns a real output vector
        @qml.qnode(qml.device(self.device, wires=self.wires), interface="torch")
        def circuit(latent, weights):
            # Encode latent vector into qubits using RX rotations
            for i in range(min(latent.shape[-1], len(self.wires))):
                qml.RX(latent[..., i], wires=self.wires[i])
            # Apply parametrized ansatz
            qml.templates.RealAmplitudes(weights, wires=self.wires)
            # Measure expectation values of PauliZ on each output wire
            return [qml.expval(qml.PauliZ(w)) for w in self.wires]

        self.circuit = circuit

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        latent : torch.Tensor
            Tensor of shape (..., latent_dim) containing latent representations.

        Returns
        -------
        torch.Tensor
            Tensor of shape (..., output_dim) containing the decoded output.
        """
        if latent.shape[-1]!= self.latent_dim:
            raise ValueError(f"Expected latent dimension {self.latent_dim}, got {latent.shape[-1]}")
        batch_shape = latent.shape[:-1]
        flat_latent = latent.reshape(-1, self.latent_dim)
        flat_output = self.circuit(flat_latent, self.weights)
        output = flat_output.reshape(*batch_shape, self.output_dim)
        return output

__all__ = ["QuantumDecoder"]
