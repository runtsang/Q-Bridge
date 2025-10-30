import torch
from torch import nn
from dataclasses import dataclass
from typing import Iterable, Tuple, List, Optional

class QuantumFullyConnectedLayer(nn.Module):
    """Placeholder quantum layer that emulates a parameterised circuit.
    In a hybrid run this would compute an expectation value via a quantum
    backend; here we use a linear transform followed by a tanh non‑linearity."""
    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, thetas: torch.Tensor) -> torch.Tensor:
        # thetas: (batch, n_features)
        return torch.tanh(self.linear(thetas)).mean(dim=0, keepdim=True)

@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1
    quantum_layers: Optional[List[int]] = None  # indices of hidden layers that are quantum

class AutoencoderHybridNet(nn.Module):
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = nn.ModuleList()
        in_dim = config.input_dim
        for idx, hidden in enumerate(config.hidden_dims):
            if config.quantum_layers and idx in config.quantum_layers:
                layer = QuantumFullyConnectedLayer(in_dim)
            else:
                layer = nn.Sequential(
                    nn.Linear(in_dim, hidden),
                    nn.ReLU(),
                    nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity()
                )
            self.encoder.append(layer)
            in_dim = hidden
        self.latent = nn.Linear(in_dim, config.latent_dim)

        self.decoder = nn.ModuleList()
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            self.decoder.append(
                nn.Sequential(
                    nn.Linear(in_dim, hidden),
                    nn.ReLU(),
                    nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity()
                )
            )
            in_dim = hidden
        self.output = nn.Linear(in_dim, config.input_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.encoder:
            if isinstance(layer, QuantumFullyConnectedLayer):
                # In a real hybrid run the input would be a tensor of parameters
                x = layer(x)
            else:
                x = layer(x)
        return self.latent(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = z
        for layer in self.decoder:
            x = layer(x)
        return self.output(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)

def Autoencoder(*,
                input_dim: int,
                latent_dim: int = 32,
                hidden_dims: Tuple[int,...] = (128, 64),
                dropout: float = 0.1,
                quantum_layers: Optional[List[int]] = None) -> AutoencoderHybridNet:
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        quantum_layers=quantum_layers or []
    )
    return AutoencoderHybridNet(config)

def train_autoencoder(model: AutoencoderHybridNet,
                      data: torch.Tensor,
                      epochs: int = 100,
                      batch_size: int = 64,
                      lr: float = 1e-3,
                      weight_decay: float = 0.0,
                      device: torch.device | None = None) -> List[float]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = torch.utils.data.TensorDataset(data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: List[float] = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch, in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)

    return history

__all__ = ["Autoencoder", "AutoencoderHybridNet", "AutoencoderConfig", "train_autoencoder"]
