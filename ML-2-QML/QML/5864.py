import pennylane as qml
import pennylane.numpy as np
import torch
from torch import nn
from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class QuantumAutoencoderConfig:
    """Configuration for the hybrid quantum‑classical autoencoder."""
    input_dim: int
    latent_dim: int = 8
    hidden_dims: Tuple[int,...] = (64, 32)
    dropout: float = 0.1

class QuantumAutoencoder(nn.Module):
    """Hybrid autoencoder: quantum encoder + classical decoder."""
    def __init__(self, cfg: QuantumAutoencoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_qubits = cfg.latent_dim
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        self.qnode = qml.QNode(self._quantum_encode, self.dev, interface="torch")

        # Classical decoder mirrors the VAE decoder
        decoder_layers = []
        in_dim = cfg.latent_dim
        for hidden in reversed(cfg.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                decoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def _quantum_encode(self, x: torch.Tensor):
        # Feature map: amplitude encoding
        qml.AmplitudeEmbedding(x, wires=range(self.n_qubits))
        # Variational layers
        for _ in range(3):
            qml.StronglyEntanglingLayers(
                weights=np.random.uniform(0, 2*np.pi, (self.n_qubits, 3)),
                wires=range(self.n_qubits),
            )
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.qnode(x)
        return self.decoder(latent)

def train_quantum_autoencoder(
    model: QuantumAutoencoder,
    data: torch.Tensor,
    *,
    epochs: int = 200,
    lr: float = 1e-3,
    device: torch.device | None = None,
) -> List[float]:
    """Training loop for the hybrid autoencoder."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    history: List[float] = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for i in range(len(data)):
            x = data[i].to(device)
            optimizer.zero_grad()
            recon = model(x)
            loss = loss_fn(recon, x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(data)
        history.append(epoch_loss)

    return history

__all__ = ["QuantumAutoencoderConfig", "QuantumAutoencoder", "train_quantum_autoencoder"]
