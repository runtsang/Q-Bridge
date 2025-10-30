import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from typing import Tuple, Iterable, List

from qml import QuantumLatent

__all__ = [
    "AutoencoderConfig",
    "AutoencoderNet",
    "HybridAutoencoder",
    "Autoencoder",
    "train_pretrain",
    "train_finetune",
]


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
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    batch_norm: bool = False
    skip_connection: bool = False
    qnn_depth: int = 2
    qnn_device: str = "default.qubit"


def _build_mlp(
    in_dim: int,
    hidden_dims: Tuple[int,...],
    out_dim: int,
    dropout: float = 0.0,
    batch_norm: bool = False,
    skip_connection: bool = False,
) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_dim_cur = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(in_dim_cur, h))
        if batch_norm:
            layers.append(nn.BatchNorm1d(h))
        layers.append(nn.ReLU())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        in_dim_cur = h
    layers.append(nn.Linear(in_dim_cur, out_dim))
    if skip_connection:
        layers.append(nn.Identity())
    return nn.Sequential(*layers)


class AutoencoderNet(nn.Module):
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.encoder = _build_mlp(
            in_dim=config.input_dim,
            hidden_dims=config.hidden_dims,
            out_dim=config.latent_dim,
            dropout=config.dropout,
            batch_norm=config.batch_norm,
            skip_connection=config.skip_connection,
        )
        self.decoder = _build_mlp(
            in_dim=config.latent_dim,
            hidden_dims=config.hidden_dims[::-1],
            out_dim=config.input_dim,
            dropout=config.dropout,
            batch_norm=config.batch_norm,
            skip_connection=config.skip_connection,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


class HybridAutoencoder(nn.Module):
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        self.classical = AutoencoderNet(config)
        self.quantum = QuantumLatent(
            latent_dim=config.latent_dim,
            depth=config.qnn_depth,
            device=config.qnn_device,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.classical.encode(x)

    def latent(self, x: torch.Tensor) -> torch.Tensor:
        return self.quantum(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.classical.decode(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_class = self.encode(x)
        z_quant = self.latent(z_class)
        return self.decode(z_quant)


def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    batch_norm: bool = False,
    skip_connection: bool = False,
    qnn_depth: int = 2,
    qnn_device: str = "default.qubit",
) -> HybridAutoencoder:
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        batch_norm=batch_norm,
        skip_connection=skip_connection,
        qnn_depth=qnn_depth,
        qnn_device=qnn_device,
    )
    return HybridAutoencoder(config)


def train_pretrain(
    model: HybridAutoencoder,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> List[float]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Freeze quantum part
    model.quantum.eval()
    for p in model.quantum.parameters():
        p.requires_grad = False

    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(
        list(model.classical.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    loss_fn = nn.MSELoss()
    history: List[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history


def train_finetune(
    model: HybridAutoencoder,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr_classical: float = 1e-3,
    lr_quantum: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> List[float]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    opt_classical = torch.optim.Adam(
        list(model.classical.parameters()),
        lr=lr_classical,
        weight_decay=weight_decay,
    )
    opt_quantum = torch.optim.Adam(
        list(model.quantum.parameters()),
        lr=lr_quantum,
        weight_decay=weight_decay,
    )
    loss_fn = nn.MSELoss()
    history: List[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            opt_classical.zero_grad()
            opt_quantum.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            opt_classical.step()
            opt_quantum.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history
