import torch
from torch import nn
from dataclasses import dataclass
from typing import Tuple, Iterable, List

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
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
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1
    batch_norm: bool = False

class Autoencoder__gen285(nn.Module):
    def __init__(self, config: AutoencoderConfig):
        super().__init__()
        self.config = config
        encoder_layers = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            if config.batch_norm:
                encoder_layers.append(nn.BatchNorm1d(hidden))
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
            if config.batch_norm:
                decoder_layers.append(nn.BatchNorm1d(hidden))
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
        return self.decode(self.encode(x))

    def reconstruction_loss(self, x: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
        return nn.functional.mse_loss(recon, x, reduction='mean')

def train_autoencoder(
    model: Autoencoder__gen285,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
    early_stop_patience: int | None = None,
) -> List[float]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = torch.utils.data.TensorDataset(_as_tensor(data))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history = []

    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch, in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)

        if early_stop_patience is not None:
            if epoch_loss < best_loss - 1e-5:
                best_loss = epoch_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stop_patience:
                    break
    return history

__all__ = ["Autoencoder__gen285", "AutoencoderConfig", "train_autoencoder"]
