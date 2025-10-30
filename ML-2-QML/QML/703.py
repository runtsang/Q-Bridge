import pennylane as qml
import pennylane.numpy as np
import torch
from torch import nn

def quantum_latent_circuit(
    num_qubits: int,
    latent_dim: int,
    weight_params: torch.Tensor,
    device: qml.Device | None = None,
) -> qml.QNode:
    """
    Returns a QNode that encodes a classical latent vector into quantum
    amplitudes and outputs a latent vector of size ``num_qubits`` via Z
    expectation values.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    latent_dim : int
        Dimension of the classical latent vector (must be <= num_qubits).
    weight_params : torch.Tensor
        Trainable variational parameters of shape (num_layers, num_qubits).
    device : qml.Device, optional
        Pennylane device; defaults to the ``default.qubit`` simulator.
    """
    if device is None:
        device = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(device, interface="torch", diff_method="backprop")
    def circuit(latent_vector: torch.Tensor):
        # Angle encoding of the classical latent vector
        for i in range(latent_dim):
            qml.RY(latent_vector[i], wires=i)

        # Variational layers (basic entangler)
        qml.apply(qml.templates.BasicEntanglerLayers(weight_params, wires=range(num_qubits)))

        # Expectation values of PauliZ on each qubit
        return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

    return circuit


class QuantumLatentEncoder(nn.Module):
    """
    Wraps the quantum circuit into a PyTorch module so that its parameters
    can be optimized together with the classical network.
    """
    def __init__(self, num_qubits: int, latent_dim: int, num_layers: int = 2):
        super().__init__()
        self.num_qubits = num_qubits
        self.latent_dim = latent_dim
        # Trainable variational parameters
        self.weight_params = nn.Parameter(torch.randn(num_layers, num_qubits))
        self.circuit = quantum_latent_circuit(num_qubits, latent_dim, self.weight_params)

    def forward(self, latent_vector: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the quantum circuit.

        Parameters
        ----------
        latent_vector : torch.Tensor
            Classical latent vector of shape (latent_dim,).

        Returns
        -------
        torch.Tensor
            Quantum latent vector of shape (num_qubits,).
        """
        # The QNode expects a single sample; for batching, apply manually
        return self.circuit(latent_vector)


__all__ = ["QuantumLatentEncoder", "quantum_latent_circuit"]
