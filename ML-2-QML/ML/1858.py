import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from typing import Tuple, Iterable

@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    kl_weight: float = 0.0
    cluster_dim: int = 0
    shape_mask: bool = False

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

class AutoencoderGen(nn.Module):
    def __init__(self, config: AutoencoderConfig, quantum_encoder=None) -> None:
        super().__init__()
        self.config = config
        self.quantum_encoder = quantum_encoder
        if quantum_encoder is None:
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
        else:
            self.encoder = quantum_encoder

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

    def reconstruction_loss(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return nn.functional.mse_loss(recon, target, reduction="sum")

    def kl_divergence(self, z: torch.Tensor) -> torch.Tensor:
        mu = z.mean(dim=0)
        var = z.var(dim=0, unbiased=False)
        return -0.5 * torch.sum(1 + torch.log(var + 1e-12) - mu.pow(2) - var)

    def cluster_loss(self, z: torch.Tensor) -> torch.Tensor:
        if self.config.cluster_dim <= 1:
            return torch.tensor(0.0, device=z.device)
        pairwise = torch.cdist(z, z, p=2)
        return pairwise.mean()

def train_autoencoder_gen(
    model: AutoencoderGen,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    kl_weight: float = 0.0,
    cluster_weight: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history: list[float] = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = model.reconstruction_loss(recon, batch)
            if kl_weight > 0.0:
                loss += kl_weight * model.kl_divergence(recon)
            if cluster_weight > 0.0:
                loss += cluster_weight * model.cluster_loss(recon)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

__all__ = ["AutoencoderGen", "AutoencoderConfig", "train_autoencoder_gen"]
