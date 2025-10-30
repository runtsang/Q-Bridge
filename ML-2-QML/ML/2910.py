import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from typing import Tuple, Iterable, Callable, Optional

# ------------------------------------------------------------------
# Classical Auto‑Encoder
# ------------------------------------------------------------------
@dataclass
class AutoencoderConfig:
    """Configuration for the hybrid auto‑encoder."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    # Size of the latent space that will be fed to the quantum circuit
    quantum_latent_size: Optional[int] = None


class AutoencoderNet(nn.Module):
    """Fully‑connected auto‑encoder with an optional quantum refinement step."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = self._build_mlp(
            in_dim=config.input_dim,
            hidden_dims=config.hidden_dims,
            out_dim=config.latent_dim,
        )
        self.decoder = self._build_mlp(
            in_dim=config.latent_dim,
            hidden_dims=config.hidden_dims[::-1],
            out_dim=config.input_dim,
        )
        # Placeholder for the quantum refinement callable
        self.quantum_refine: Optional[Callable[[torch.Tensor], torch.Tensor]] = None

    def _build_mlp(self, in_dim: int, hidden_dims: Tuple[int,...], out_dim: int) -> nn.Sequential:
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if self.config.dropout > 0.0:
                layers.append(nn.Dropout(self.config.dropout))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        return nn.Sequential(*layers)

    def set_quantum_refine(self, refine_fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
        """Attach a quantum refinement callable that will be applied to the latent code."""
        self.quantum_refine = refine_fn

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        if self.quantum_refine is not None:
            z = self.quantum_refine(z)
        return self.decode(z)


def train_autoencoder(
    model: AutoencoderNet,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: Optional[torch.device] = None,
) -> list[float]:
    """Training loop that supports a quantum refinement step."""
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
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history


def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Ensure input is a float32 tensor on the default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor


__all__ = ["AutoencoderConfig", "AutoencoderNet", "train_autoencoder"]
