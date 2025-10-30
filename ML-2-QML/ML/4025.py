"""Hybrid classical–quantum autoencoder implementation.

The module exposes a `HybridAutoencoder` class that contains:
  * A classical MLP encoder/decoder (same as the seed Autoencoder).
  * An optional quantum encoder that maps classical inputs to a quantum
    state using TorchQuantum.  The quantum decoder is approximated by a
    classical linear layer, so the output remains a tensor.
  * A training helper that supports both modes.

The design follows the 'combination' scaling paradigm, merging ideas from
the two seed projects: the dense autoencoder architecture and the quantum
regression encoder.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Tuple, List, Optional

import torch
from torch import nn
import torchquantum as tq


@dataclass
class AutoencoderConfig:
    """Configuration for the hybrid autoencoder."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    quantum: bool = False  # whether to use the quantum encoder


class ClassicalAutoencoder(nn.Module):
    """A pure‑classical MLP autoencoder."""

    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.encoder = self._build_mlp(
            config.input_dim, config.latent_dim, config.hidden_dims, config.dropout
        )
        self.decoder = self._build_mlp(
            config.latent_dim, config.input_dim, config.hidden_dims[::-1], config.dropout
        )

    @staticmethod
    def _build_mlp(
        in_dim: int,
        out_dim: int,
        hidden_dims: Tuple[int, int],
        dropout: float,
    ) -> nn.Sequential:
        layers: List[nn.Module] = []
        prev = in_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev = hidden
        layers.append(nn.Linear(prev, out_dim))
        return nn.Sequential(*layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(x))


class QuantumEncoder(tq.QuantumModule):
    """Quantum encoder that lifts a classical vector to a quantum state."""

    class _QLayer(tq.QuantumModule):
        def __init__(self, num_wires: int) -> None:
            super().__init__()
            self.num_wires = num_wires
            # Random layer + local rotations
            self.random_layer = tq.RandomLayer(n_ops=20, wires=list(range(num_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            for w in range(self.num_wires):
                self.rx(qdev, wires=w)
                self.ry(qdev, wires=w)

    def __init__(self, num_wires: int) -> None:
        super().__init__()
        self.num_wires = num_wires
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict[f"{num_wires}xRy"]
        )
        self.layer = self._QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """Encode a batch of classical vectors into a quantum state."""
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(
            n_wires=self.num_wires,
            bsz=bsz,
            device=state_batch.device,
        )
        self.encoder(qdev, state_batch)
        self.layer(qdev)
        return self.measure(qdev)


class HybridAutoencoder(nn.Module):
    """Hybrid classical–quantum autoencoder.

    The encoder can be classical or quantum (controlled by the `quantum`
    flag in `AutoencoderConfig`).  The decoder is always classical.
    """

    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        self.classical = ClassicalAutoencoder(config)
        self.quantum = QuantumEncoder(config.latent_dim) if config.quantum else None

    def encode(self, x: torch.Tensor, *, use_quantum: bool = False) -> torch.Tensor:
        if use_quantum and self.quantum:
            return self.quantum(x)
        return self.classical.encode(x)

    def decode(self, z: torch.Tensor, *, use_quantum: bool = False) -> torch.Tensor:
        # For simplicity, use the classical decoder even when the
        # quantum encoder was used.  This keeps the overall network
        # differentiable and trainable with standard optimizers.
        return self.classical.decode(z)

    def forward(self, x: torch.Tensor, *, use_quantum: bool = False) -> torch.Tensor:  # type: ignore[override]
        latent = self.encode(x, use_quantum=use_quantum)
        return self.decode(latent, use_quantum=use_quantum)


def HybridAutoencoderFactory(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    quantum: bool = False,
) -> HybridAutoencoder:
    """Convenience factory mirroring the original `Autoencoder` helper."""
    cfg = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        quantum=quantum,
    )
    return HybridAutoencoder(cfg)


def train_hybrid_autoencoder(
    model: HybridAutoencoder,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: Optional[torch.device] = None,
    use_quantum: bool = False,
) -> List[float]:
    """Simple training loop for the hybrid autoencoder."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = torch.utils.data.TensorDataset(data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: List[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon = model(batch, use_quantum=use_quantum)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)

    return history


__all__ = [
    "AutoencoderConfig",
    "ClassicalAutoencoder",
    "QuantumEncoder",
    "HybridAutoencoder",
    "HybridAutoencoderFactory",
    "train_hybrid_autoencoder",
]
