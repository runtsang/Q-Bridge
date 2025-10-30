import pennylane as qml
import torch
import numpy as np

class QuantumKernel(torch.nn.Module):
    """Quantum kernel that maps 4‑dim classical patches to a 4‑bit measurement."""
    def __init__(self, n_wires: int = 4, n_layers: int = 4, device: str = "default.qubit") -> None:
        super().__init__()
        self.n_wires = n_wires
        self.n_layers = n_layers
        self.dev = qml.device(device, wires=n_wires)
        # Trainable parameters for the random layers
        self.weights = torch.nn.Parameter(torch.randn(n_layers, n_wires))

        @qml.qnode(self.dev, interface="torch")
        def circuit(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
            # Encode classical data into rotation angles
            for i in range(n_wires):
                qml.RY(x[i], wires=i)
            # Random layers with trainable angles
            for layer in range(n_layers):
                for w in range(n_wires):
                    qml.RZ(weights[layer, w], wires=w)
                for w in range(n_wires - 1):
                    qml.CNOT(wires=[w, w + 1])
            # Measure Pauli‑Z on each wire
            return [qml.expval(qml.PauliZ(i)) for i in range(n_wires)]

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Batch of 4‑dim patches, shape (batch, 4).

        Returns
        -------
        torch.Tensor
            Quantum measurements, shape (batch, 4).
        """
        return self.circuit(x, self.weights)

__all__ = ["QuantumKernel"]
