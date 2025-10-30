"""Quantum‑classical autoencoder using Pennylane.

The encoder is a variational RealAmplitudes circuit that maps input features to a latent
representation. The decoder is a small fully‑connected network that reconstructs the
original data. Training uses Pennylane’s parameter‑shift gradients and Adam on the
combined quantum‑classical parameters.
"""

import pennylane as qml
import torch
from torch import nn
import numpy as np

class Autoencoder:
    """Variational quantum autoencoder with a classical decoder."""
    def __init__(self,
                 input_dim: int,
                 latent_dim: int | None = None,
                 hidden_dims: tuple[int,...] = (32,),
                 num_qubits: int | None = None,
                 device: str = "cpu") -> None:
        self.input_dim = input_dim
        self.num_qubits = num_qubits or int(np.ceil(np.log2(input_dim)))
        self.latent_dim = latent_dim or self.num_qubits
        if self.latent_dim > self.num_qubits:
            raise ValueError("latent_dim cannot exceed number of qubits")
        self.dev = qml.device("default.qubit", wires=self.num_qubits)
        self.encoder_params = nn.Parameter(torch.randn(self.num_qubits * 3))
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], input_dim)
        )
        self.device = device
        self._build_qnode()

    def _build_qnode(self) -> None:
        @qml.qnode(self.dev, interface="torch", diff_method="parameter-shift")
        def circuit(x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
            qml.templates.AngleEmbedding(x, wires=range(self.num_qubits))
            qml.templates.RealAmplitudes(params, wires=range(self.num_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.latent_dim)]
        self.circuit = circuit

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Map classical input to a latent vector via the quantum circuit."""
        return self.circuit(x, self.encoder_params)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct the input from the latent vector."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """End‑to‑end forward pass."""
        z = self.encode(x)
        return self.decode(z)

    def train(self,
              data: torch.Tensor,
              epochs: int = 200,
              lr: float = 1e-3,
              batch_size: int = 32,
              weight_decay: float = 0.0) -> list[float]:
        """Train the hybrid autoencoder end‑to‑end."""
        optimizer = torch.optim.Adam(
            list(self.encoder_params) + list(self.decoder.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )
        loss_fn = nn.MSELoss()
        history: list[float] = []

        dataset = torch.utils.data.TensorDataset(data)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch, in loader:
                optimizer.zero_grad()
                recon = self.forward(batch)
                loss = loss_fn(recon, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch.size(0)
            epoch_loss /= len(dataset)
            history.append(epoch_loss)
        return history

__all__ = ["Autoencoder"]
