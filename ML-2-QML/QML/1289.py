"""Quantum autoencoder built with PennyLane.

The class `Autoencoder` wraps a variational circuit that embeds the
input as angles, compresses it into a latent subspace, and reconstructs
the original vector via a measurement of the hidden qubits.  The
circuit uses a RealAmplitudes ansatz with two repetitions and
supports a custom device.  Training is performed with a simple
gradient‑based optimiser and returns a loss history.
"""

import pennylane as qml
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Iterable, Tuple, Dict, Optional

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    return tensor.to(dtype=torch.float32)

class Autoencoder(nn.Module):
    """Variational quantum autoencoder using PennyLane."""
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 3,
        num_qubits: Optional[int] = None,
        device: str = "default.qubit",
        wires: Optional[Tuple[int,...]] = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_qubits = num_qubits or input_dim
        self.device = device
        self.wires = wires or tuple(range(self.num_qubits))

        self.dev = qml.device(self.device, wires=self.wires)

        # Parameters for the variational ansatz
        self.ansatz_params = nn.Parameter(
            torch.randn((2, self.num_qubits))
        )  # two repetitions of RealAmplitudes

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(inputs: torch.Tensor) -> torch.Tensor:
            # Feature embedding
            qml.AngleEmbedding(inputs, wires=self.wires)
            # Variational layers
            for rep in range(2):
                qml.RealAmplitudes(self.ansatz_params[rep], wires=self.wires)
            # Measurement of latent qubits
            return qml.expval(qml.PauliZ(self.wires[: self.latent_dim]))

        self.circuit = circuit

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.circuit(inputs)

def AutoencoderFactory(
    input_dim: int,
    *,
    latent_dim: int = 3,
    num_qubits: Optional[int] = None,
    device: str = "default.qubit",
) -> Autoencoder:
    return Autoencoder(input_dim, latent_dim=latent_dim, num_qubits=num_qubits, device=device)

def train_autoencoder(
    model: Autoencoder,
    data: torch.Tensor,
    *,
    epochs: int = 200,
    batch_size: int = 32,
    lr: float = 1e-2,
    device: Optional[torch.device] = None,
    early_stopping_patience: int = 10,
) -> Dict[str, list[float]]:
    """Training loop for the quantum autoencoder."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    history: Dict[str, list[float]] = {"train_loss": []}
    best_val = float("inf")
    patience = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            # Target is the latent‑subspace representation of the input
            loss = loss_fn(recon, batch[:, :model.latent_dim])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history["train_loss"].append(epoch_loss)

        if epoch_loss < best_val:
            best_val = epoch_loss
            patience = 0
        else:
            patience += 1
            if patience >= early_stopping_patience:
                break
    return history

__all__ = ["Autoencoder", "AutoencoderFactory", "train_autoencoder"]
