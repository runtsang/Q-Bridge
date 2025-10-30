import torch
import torch.nn as nn
import pennylane as qml
import pennylane.numpy as np

class QuantumNATGen257(nn.Module):
    """
    Quantum encoder for the Quantum‑NAT architecture.
    Uses a parameterised rotation encoder followed by a 2‑layer
    variational circuit. The output is a 4‑dimensional feature vector
    produced from the expectation values of Pauli‑Z on each qubit.
    """

    def __init__(self, n_wires: int = 4, n_layers: int = 2,
                 device: str = "default.qubit") -> None:
        super().__init__()
        self.n_wires = n_wires
        self.n_layers = n_layers
        self.device = qml.device(device, wires=n_wires)
        # Trainable parameters of the variational circuit
        self.weights = nn.Parameter(torch.randn(n_layers, n_wires, 3))
        # Batch normalization on the quantum output
        self.bn = nn.BatchNorm1d(n_wires)

        # Define the QNode
        @qml.qnode(self.device, interface="torch", diff_method="backprop")
        def circuit(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
            # Encode classical inputs via Ry rotations
            for i in range(n_wires):
                qml.RY(inputs[i], wires=i)
            # Variational layers
            for layer in range(n_layers):
                for wire in range(n_wires):
                    qml.Rot(weights[layer, wire, 0],
                            weights[layer, wire, 1],
                            weights[layer, wire, 2],
                            wires=wire)
                # Entangling CNOT chain
                for wire in range(n_wires - 1):
                    qml.CNOT(wires=[wire, wire + 1])
            # Measure expectation values of PauliZ
            return [qml.expval(qml.PauliZ(w)) for w in range(n_wires)]

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, n_wires)

        Returns
        -------
        torch.Tensor
            Normalised 4‑dimensional quantum feature vector
        """
        # Process each sample in the batch
        batch_out = torch.stack([self.circuit(sample, self.weights)
                                 for sample in x])
        return self.bn(batch_out)

__all__ = ["QuantumNATGen257"]
