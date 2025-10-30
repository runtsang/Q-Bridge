"""Hybrid quantum‑classical autoencoder implemented with Pennylane."""
import pennylane as qml
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from typing import Tuple, List, Optional

def _as_tensor(data: torch.Tensor | List[float]) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 16
    hidden_dims: Tuple[int,...] = (32, 16)
    n_layers: int = 3
    beta: float = 1.0

class AutoencoderHybrid(nn.Module):
    def __init__(self, config: AutoencoderConfig):
        super().__init__()
        self.config = config
        # Classical encoder
        encoder_layers = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.extend([nn.Linear(in_dim, hidden), nn.ReLU()])
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        # Quantum decoder
        self.dev = qml.device("default.qubit", wires=config.latent_dim)
        self.qnode = qml.QNode(self._quantum_decoder, self.dev, interface="torch")
        self.qnn_params = nn.Parameter(torch.randn(config.n_layers, config.latent_dim, 3))
        # Map qubit expectations to reconstruction
        self.output_layer = nn.Linear(config.latent_dim, config.input_dim)

    def _quantum_decoder(self, z: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        for i in range(self.config.latent_dim):
            qml.RY(z[i], wires=i)
        qml.StronglyEntanglingLayers(params, wires=range(self.config.latent_dim))
        return [qml.expval(qml.PauliZ(i)) for i in range(self.config.latent_dim)]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        q_output = self.qnode(z, self.qnn_params)
        return self.output_layer(q_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)

    def loss(self, recon: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.mse_loss(recon, x, reduction='sum')

def train_autoencoder(
    model: AutoencoderHybrid,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: Optional[torch.device] = None,
    early_stopping_patience: Optional[int] = None,
) -> List[float]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history: List[float] = []
    best_loss = float("inf")
    patience = 0
    for epoch in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon = model(batch)
            loss = model.loss(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
        if early_stopping_patience is not None:
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience = 0
            else:
                patience += 1
                if patience >= early_stopping_patience:
                    break
    return history

__all__ = ["AutoencoderHybrid", "AutoencoderConfig", "train_autoencoder"]
