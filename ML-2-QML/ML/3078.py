import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from typing import Iterable, Tuple, Sequence

class RBFKernel(nn.Module):
    """Classical radial‑basis‑function kernel with optional learnable width."""
    def __init__(self, gamma: float = 1.0, trainable: bool = False):
        super().__init__()
        if trainable:
            self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))
        else:
            self.gamma = torch.tensor(gamma, dtype=torch.float32)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff ** 2, dim=-1, keepdim=True))

@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

class AutoencoderNet(nn.Module):
    """Fully‑connected autoencoder with symmetric encoder/decoder."""
    def __init__(self, config: AutoencoderConfig):
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
    """Factory returning a configured autoencoder."""
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    return AutoencoderNet(config)

class UnifiedKernelAutoencoder(nn.Module):
    """
    Hybrid kernel‑autoencoder that exposes both classical RBF kernels
    and a latent‑space kernel derived from the encoder.
    """
    def __init__(
        self,
        config: AutoencoderConfig,
        gamma: float = 1.0,
        kernel_trainable: bool = False,
        latent_gamma: float = 1.0,
        latent_trainable: bool = False,
    ):
        super().__init__()
        self.autoencoder = AutoencoderNet(config)
        self.classical_kernel = RBFKernel(gamma, trainable=kernel_trainable)
        self.latent_kernel = RBFKernel(latent_gamma, trainable=latent_trainable)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.encode(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.decode(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(x))

    def compute_classical_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.classical_kernel(x, y)

    def compute_latent_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        z_x = self.encode(x)
        z_y = self.encode(y)
        return self.latent_kernel(z_x, z_y)

    def compute_combined_kernel(
        self, x: torch.Tensor, y: torch.Tensor, alpha: float = 0.5
    ) -> torch.Tensor:
        return alpha * self.compute_classical_kernel(x, y) + (1 - alpha) * self.compute_latent_kernel(x, y)

    def kernel_matrix(
        self,
        a: Sequence[torch.Tensor],
        b: Sequence[torch.Tensor],
        alpha: float = 0.5,
    ) -> np.ndarray:
        """Compute Gram matrix for sequences a and b."""
        mat = np.empty((len(a), len(b)), dtype=np.float64)
        for i, xi in enumerate(a):
            for j, yj in enumerate(b):
                mat[i, j] = self.compute_combined_kernel(xi, yj, alpha).item()
        return mat

def train_autoencoder(
    model: nn.Module,
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
    """Return a float32 tensor on the current default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

__all__ = [
    "RBFKernel",
    "AutoencoderConfig",
    "AutoencoderNet",
    "Autoencoder",
    "UnifiedKernelAutoencoder",
    "train_autoencoder",
]
