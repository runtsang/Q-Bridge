import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from typing import Iterable, Tuple, List, Optional

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Ensure the input is a float32 tensor on the current default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

@dataclass
class AutoencoderConfig:
    """Configuration values for :class:`Autoencoder__gen013`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    early_stop_patience: int = 10

class Autoencoder__gen013(nn.Module):
    """Fully‑connected autoencoder with optional early‑stopping."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.encoder = self._build_mlp(
            config.input_dim, config.hidden_dims, config.latent_dim, config.dropout
        )
        self.decoder = self._build_mlp(
            config.latent_dim,
            tuple(reversed(config.hidden_dims)),
            config.input_dim,
            config.dropout,
        )

    def _build_mlp(
        self,
        in_dim: int,
        hidden_dims: Tuple[int,...],
        out_dim: int,
        dropout: float,
    ) -> nn.Sequential:
        layers: List[nn.Module] = []
        for hidden in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, out_dim))
        return nn.Sequential(*layers)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(inputs))

def train_autoencoder(
    model: Autoencoder__gen013,
    train_data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: Optional[torch.device] = None,
    val_data: Optional[torch.Tensor] = None,
    patience: int = 10,
) -> dict[str, List[float]]:
    """Training loop that supports early‑stopping on a validation set."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader = DataLoader(
        TensorDataset(_as_tensor(train_data)), batch_size=batch_size, shuffle=True
    )
    if val_data is not None:
        val_loader = DataLoader(
            TensorDataset(_as_tensor(val_data)), batch_size=batch_size, shuffle=False
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history = {"train_loss": [], "val_loss": []}

    best_val = float("inf")
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(train_loader.dataset)
        history["train_loss"].append(epoch_loss)

        if val_data is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for (batch,) in val_loader:
                    batch = batch.to(device)
                    recon = model(batch)
                    loss = loss_fn(recon, batch)
                    val_loss += loss.item() * batch.size(0)
            val_loss /= len(val_loader.dataset)
            history["val_loss"].append(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    return history

__all__ = ["Autoencoder__gen013", "AutoencoderConfig", "train_autoencoder"]
