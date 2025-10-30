import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from typing import Tuple, Iterable, Optional

@dataclass
class AutoencoderGen280Config:
    """
    Configuration for the hybrid autoencoder.
    """
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    quantum_encoder: bool = False
    qdevice: str = "cpu"

class QuantumEncoder(nn.Module):
    """
    Placeholder for a quantum encoder that can be swapped with a real QNN.
    """
    def __init__(self, latent_dim: int, qdevice: str = "cpu"):
        super().__init__()
        self.latent_dim = latent_dim
        self.qdevice = qdevice
        self.linear = nn.Linear(0, latent_dim)  # will be overridden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.size(0), self.latent_dim, device=x.device)

class AutoencoderGen280(nn.Module):
    """
    Hybrid autoencoder that optionally uses a quantum encoder.
    """
    def __init__(self, config: AutoencoderGen280Config, quantum_encoder: Optional[nn.Module] = None):
        super().__init__()
        self.config = config
        if config.quantum_encoder:
            if quantum_encoder is None:
                raise ValueError("quantum_encoder must be provided when quantum_encoder=True")
            self.encoder = quantum_encoder
        else:
            encoder_layers = []
            in_dim = config.input_dim
            for hidden in config.hidden_dims:
                encoder_layers.append(nn.Linear(in_dim, hidden))
                encoder_layers.append(nn.ReLU())
                if config.dropout > 0.0:
                    encoder_layers.append(nn.Dropout(config.dropout))
                in_dim = hidden
            encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
            self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)

def AutoencoderGen280Factory(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    quantum_encoder: Optional[nn.Module] = None,
) -> AutoencoderGen280:
    """
    Factory that returns a hybrid autoencoder model.
    """
    config = AutoencoderGen280Config(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        quantum_encoder=quantum_encoder is not None,
    )
    return AutoencoderGen280(config, quantum_encoder=quantum_encoder)

def train_autoencoder_gen280(
    model: AutoencoderGen280,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """
    Simple reconstruction training loop that jointly optimises the encoder and decoder.
    If a quantum encoder is supplied it is treated as a trainable PyTorch module.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: list[float] = []

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
    return history

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """
    Utility that converts an iterable or tensor into a float32 tensor.
    """
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

__all__ = [
    "AutoencoderGen280",
    "AutoencoderGen280Factory",
    "AutoencoderGen280Config",
    "QuantumEncoder",
    "train_autoencoder_gen280",
]
