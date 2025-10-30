import torch
from torch import nn
from dataclasses import dataclass
from typing import Tuple, List, Iterable

from torch.utils.data import DataLoader, TensorDataset

# Import quantum helper
from qml_code import create_sampler_qnn


@dataclass
class UnifiedAutoEncoderConfig:
    """Hyper‑parameters for the hybrid auto‑encoder."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    # Quantum parameters
    q_nqubits: int | None = None  # if None, defaults to latent_dim
    q_reps: int = 5
    q_shift: float = 0.0


class UnifiedAutoEncoderNet(nn.Module):
    """Hybrid classical‑quantum auto‑encoder."""

    def __init__(self, cfg: UnifiedAutoEncoderConfig):
        super().__init__()
        self.cfg = cfg
        # Default to latent_dim qubits if not provided
        if cfg.q_nqubits is None:
            cfg.q_nqubits = cfg.latent_dim

        # Encoder
        enc_layers: List[nn.Module] = []
        dim = cfg.input_dim
        for h in cfg.hidden_dims:
            enc_layers.append(nn.Linear(dim, h))
            enc_layers.append(nn.ReLU())
            if cfg.dropout > 0:
                enc_layers.append(nn.Dropout(cfg.dropout))
            dim = h
        enc_layers.append(nn.Linear(dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Optional quantum layer
        if cfg.q_nqubits == 0:
            self.quantum = None
        else:
            self.quantum = create_sampler_qnn(
                num_qubits=cfg.q_nqubits,
                reps=cfg.q_reps,
                entanglement="circular",
                output_shape=1,
            )

        # Decoder
        dec_input_dim = cfg.latent_dim
        if self.quantum is not None:
            dec_input_dim += 1  # quantum scalar output
        dec_layers: List[nn.Module] = []
        dim = dec_input_dim
        for h in reversed(cfg.hidden_dims):
            dec_layers.append(nn.Linear(dim, h))
            dec_layers.append(nn.ReLU())
            if cfg.dropout > 0:
                dec_layers.append(nn.Dropout(cfg.dropout))
            dim = h
        dec_layers.append(nn.Linear(dim, cfg.input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def quantum_forward(self, z: torch.Tensor) -> torch.Tensor:
        if self.quantum is None:
            return torch.zeros(z.size(0), 1, device=z.device, dtype=z.dtype)
        return self.quantum(z).float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        qz = self.quantum_forward(z)
        if self.quantum is not None:
            z = torch.cat([z, qz], dim=1)
        return self.decoder(z)


def UnifiedAutoEncoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    q_nqubits: int | None = None,
    q_reps: int = 5,
    q_shift: float = 0.0,
) -> UnifiedAutoEncoderNet:
    cfg = UnifiedAutoEncoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        q_nqubits=q_nqubits,
        q_reps=q_reps,
        q_shift=q_shift,
    )
    return UnifiedAutoEncoderNet(cfg)


def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor


def train_autoencoder(
    model: UnifiedAutoEncoderNet,
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
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history


__all__ = [
    "UnifiedAutoEncoderConfig",
    "UnifiedAutoEncoderNet",
    "UnifiedAutoEncoder",
    "train_autoencoder",
]
