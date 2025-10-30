"""Hybrid autoencoder combining a Pennylane quantum encoder and a classical decoder.

The class `HybridAutoencoder` inherits from `torch.nn.Module` and uses a
`qml.QNode` to compute a latent representation from the input features.
The latent vector is then decoded by a small fully‑connected network.
The model can be trained end‑to‑end with PyTorch optimisers; Pennylane
provides automatic gradients for the quantum part via the parameter‑shift rule.
"""

from __future__ import annotations

import pennylane as qml
import torch
from torch import nn
from typing import Tuple, Iterable

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Return a float32 tensor on the current default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor


class HybridAutoencoder(nn.Module):
    """Quantum‑classical variational auto‑encoder."""
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 8,
        hidden_dims: Tuple[int, int] = (64, 32),
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.device = device

        # Quantum device
        self.q_device = qml.device("default.qubit", wires=latent_dim)

        # Quantum encoder parameters
        self.encoder_params = nn.Parameter(
            torch.randn(latent_dim), requires_grad=True
        )

        # Quantum encoder as a QNode
        @qml.qnode(self.q_device, interface="torch", diff_method="parameter-shift")
        def quantum_encoder(x: torch.Tensor) -> torch.Tensor:
            # Feature map: rotate each qubit by the input value
            qml.templates.AngleEmbedding(x, wires=range(latent_dim))
            # Variational layer
            qml.templates.RealAmplitudes(self.encoder_params, wires=range(latent_dim))
            # Return expectation values of PauliZ as latent vector
            return [qml.expval(qml.PauliZ(i)) for i in range(latent_dim)]

        self.quantum_encoder = quantum_encoder

        # Classical decoder
        decoder_layers = []
        in_dim = latent_dim
        for hidden in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode with a quantum circuit and decode classically."""
        # Compute latent vector via quantum circuit
        z = self.quantum_encoder(x.to(self.device))
        # Decode to reconstruct input
        recon = self.decoder(z)
        return recon


def train_hybrid_autoencoder_qml(
    model: HybridAutoencoder,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """End‑to‑end training loop for the quantum‑classical auto‑encoder."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = torch.utils.data.TensorDataset(_as_tensor(data))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history: list[float] = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon = model(batch)
            loss = nn.functional.mse_loss(recon, batch, reduction="sum")
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history


__all__ = [
    "HybridAutoencoder",
    "train_hybrid_autoencoder_qml",
]
