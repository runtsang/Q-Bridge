import numpy as np
import torch
from torch import nn
from dataclasses import dataclass
from typing import Iterable, Tuple, Sequence

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Return a float32 tensor on the current default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

class KernalAnsatz(nn.Module):
    """Compute an RBF kernel value between two 1‑D tensors."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff ** 2, dim=-1, keepdim=True))

class Kernel(nn.Module):
    """Wrapper that normalises shape and calls :class:`KernalAnsatz`."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

@dataclass
class AutoencoderConfig:
    """Configuration values for :class:`AutoencoderNet`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

class AutoencoderNet(nn.Module):
    """A lightweight multilayer perceptron autoencoder."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
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

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(inputs))

def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
) -> AutoencoderNet:
    """Factory that mirrors the quantum helper returning a configured network."""
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
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
) -> list[float]:
    """Simple reconstruction training loop returning the loss history."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = torch.utils.data.TensorDataset(_as_tensor(data))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

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

class QuantumAutoencoderKernel(nn.Module):
    """Hybrid kernel that encodes data via a classical autoencoder and then applies an RBF kernel on the latent space."""
    def __init__(self, autoencoder: AutoencoderNet, gamma: float = 1.0) -> None:
        super().__init__()
        self.autoencoder = autoencoder
        self.kernel = Kernel(gamma)
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        latent_x = self.autoencoder.encode(x)
        latent_y = self.autoencoder.encode(y)
        return self.kernel(latent_x, latent_y)

def kernel_matrix(
    a: Sequence[torch.Tensor],
    b: Sequence[torch.Tensor],
    autoencoder: AutoencoderNet,
    gamma: float = 1.0,
) -> np.ndarray:
    """Evaluate the Gram matrix between datasets ``a`` and ``b`` using the hybrid kernel."""
    kernel = QuantumAutoencoderKernel(autoencoder, gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

__all__ = [
    "KernalAnsatz",
    "Kernel",
    "AutoencoderConfig",
    "AutoencoderNet",
    "Autoencoder",
    "train_autoencoder",
    "_as_tensor",
    "QuantumAutoencoderKernel",
    "kernel_matrix",
]
