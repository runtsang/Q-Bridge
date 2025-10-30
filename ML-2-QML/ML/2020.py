import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from typing import Tuple, Iterable, Optional, List

__all__ = ["Autoencoder", "AutoencoderConfig", "AutoencoderNet", "train_autoencoder"]

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Return a float32 tensor on the current default device."""
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
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (256, 128, 64)
    depth: int = 4
    dropout: float = 0.1
    use_quantum: bool = False
    quantum_params: dict | None = None

class ResidualBlock(nn.Module):
    """A simple residual block with two linear layers."""
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.lin1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.lin2(out)
        out = self.bn2(out)
        out = self.dropout(out)
        return self.relu(out + residual)

class AutoencoderNet(nn.Module):
    """Hybrid residual autoencoder with optional quantum latent space."""
    def __init__(self, config: AutoencoderConfig):
        super().__init__()
        self.config = config
        # Encoder
        encoder_layers: List[nn.Module] = []
        in_dim = config.input_dim
        for _ in range(config.depth):
            encoder_layers.append(nn.Linear(in_dim, in_dim))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(ResidualBlock(in_dim, dropout=config.dropout))
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers: List[nn.Module] = []
        decoder_in_dim = config.latent_dim
        for _ in range(config.depth):
            decoder_layers.append(nn.Linear(decoder_in_dim, decoder_in_dim))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(ResidualBlock(decoder_in_dim, dropout=config.dropout))
        decoder_layers.append(nn.Linear(decoder_in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        if config.use_quantum and config.quantum_params:
            # Placeholder for a quantum transformation; in practice replace with a QNN.
            self.quantum_layer = nn.Linear(decoder_in_dim, decoder_in_dim)
        else:
            self.quantum_layer = None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.decoder(z)
        if self.quantum_layer is not None:
            x = self.quantum_layer(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)

def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int,...] = (256, 128, 64),
    depth: int = 4,
    dropout: float = 0.1,
    use_quantum: bool = False,
    quantum_params: dict | None = None,
) -> AutoencoderNet:
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        depth=depth,
        dropout=dropout,
        use_quantum=use_quantum,
        quantum_params=quantum_params,
    )
    return AutoencoderNet(config)

def train_autoencoder(
    model: AutoencoderNet,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> List[float]:
    """Train the hybrid autoencoder using MSE loss and a learning‑rate scheduler."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=10, factor=0.5)
    loss_fn = nn.MSELoss()
    history: List[float] = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
        scheduler.step(epoch_loss)
    return history
