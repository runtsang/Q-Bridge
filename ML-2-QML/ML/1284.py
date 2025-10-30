import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from typing import Iterable, Tuple, Callable, List, Optional

@dataclass
class HybridAutoencoderConfig:
    """Configuration for the hybrid autoencoder."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1
    residual: bool = True
    batch_norm: bool = True

class ResidualBlock(nn.Module):
    """Residual block with optional batch‑norm and dropout."""
    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim) if out_dim == in_dim else None
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        if self.bn:
            out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        if x.shape[-1] == out.shape[-1]:
            out = out + x
        return out

class HybridAutoencoder(nn.Module):
    """Hybrid classical‑quantum autoencoder."""
    def __init__(self, config: HybridAutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = self._build_encoder(config)
        self.decoder: Optional[Callable[[torch.Tensor], torch.Tensor]] = None

    def _build_encoder(self, config: HybridAutoencoderConfig) -> nn.Sequential:
        layers: List[nn.Module] = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            layers.append(ResidualBlock(in_dim, hidden, config.dropout))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, config.latent_dim))
        return nn.Sequential(*layers)

    def set_decoder(self, decoder: Callable[[torch.Tensor], torch.Tensor]) -> None:
        self.decoder = decoder

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if self.decoder is None:
            raise RuntimeError("Decoder not set. Call set_decoder() before decoding.")
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)

def train_hybrid_autoencoder(
    model: HybridAutoencoder,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
    decoder_train_func: Optional[Callable[[HybridAutoencoder, torch.Tensor], None]] = None,
) -> List[float]:
    """Train the classical encoder and optionally the quantum decoder."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: List[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction = model(batch)
            loss = loss_fn(reconstruction, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)

    if decoder_train_func is not None:
        decoder_train_func(model, data)

    return history

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

__all__ = [
    "HybridAutoencoder",
    "HybridAutoencoderConfig",
    "ResidualBlock",
    "train_hybrid_autoencoder",
    "_as_tensor",
]
