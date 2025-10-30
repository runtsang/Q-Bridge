import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from typing import Tuple, Iterable, List, Optional

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Return a float32 tensor on the current default device."""
    tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

@dataclass
class AutoencoderConfig:
    """Configuration for :class:`AutoencoderNet`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    # New: controls trade‑off between reconstruction and quantum fidelity
    fidelity_weight: float = 0.0

class AutoencoderNet(nn.Module):
    """Hybrid classical autoencoder with optional quantum regularisation."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = self._build_mlp(is_encoder=True)
        self.decoder = self._build_mlp(is_encoder=False)

    def _build_mlp(self, *, is_encoder: bool) -> nn.Sequential:
        layers: List[nn.Module] = []
        in_dim = self.config.input_dim if is_encoder else self.config.latent_dim
        hidden_dims = self.config.hidden_dims if is_encoder else tuple(reversed(self.config.hidden_dims))
        for hidden in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU())
            if self.config.dropout > 0.0:
                layers.append(nn.Dropout(self.config.dropout))
            in_dim = hidden
        # Final linear layer
        out_dim = self.config.latent_dim if is_encoder else self.config.input_dim
        layers.append(nn.Linear(in_dim, out_dim))
        return nn.Sequential(*layers)

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
    fidelity_weight: float = 0.0,
) -> AutoencoderNet:
    """Factory that mirrors the quantum helper returning a configured network."""
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        fidelity_weight=fidelity_weight,
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
    qnn: Optional[object] = None,  # expects a callable returning fidelity
) -> List[float]:
    """
    Simple reconstruction training loop returning the loss history.
    If a quantum neural network (qnn) is provided, a fidelity penalty is added.
    The qnn should accept a latent tensor and return a scalar fidelity value.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if qnn is not None:
        qnn.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + ([] if qnn is None else list(qnn.parameters())),
        lr=lr,
        weight_decay=weight_decay,
    )
    loss_fn = nn.MSELoss()
    history: List[float] = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction = model(batch)
            recon_loss = loss_fn(reconstruction, batch)
            loss = recon_loss
            if qnn is not None:
                latent = model.encode(batch)
                # Quantum fidelity is expected to be in [0,1]
                fidelity = qnn(latent).detach()
                fidelity_penalty = (1.0 - fidelity).mean()
                loss = recon_loss + model.config.fidelity_weight * fidelity_penalty
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

__all__ = ["Autoencoder", "AutoencoderConfig", "AutoencoderNet", "train_autoencoder"]
