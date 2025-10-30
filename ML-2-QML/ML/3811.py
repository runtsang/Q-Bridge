"""
Hybrid Quanvolution + Autoencoder – Classical implementation.

Combines ideas from the original quanvolution filter with a fully‑connected
autoencoder, adding dropout and configurable hidden layers.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq


class QuanvolutionFilter(tq.QuantumModule):
    """
    Apply a random two‑qubit quantum kernel to each 2×2 patch of a 28×28 image.
    The kernel is a learnable RandomLayer followed by measurement of all qubits.
    """

    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)

        # reshape to 28×28 for patch extraction
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                data = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, data)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)


class HybridQuanvolutionAutoencoder(nn.Module):
    """
    Classical autoencoder that uses a quanvolution filter as its first layer.
    Encoder: quanvolution → dense → latent.
    Decoder: dense → reconstructor (flattened to 28×28).
    """

    def __init__(
        self,
        hidden_dims: tuple[int,...] = (128, 64),
        latent_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()

        # Encoder
        encoder_layers = []
        in_dim = 4 * 14 * 14  # 4 features per 2×2 patch, 14×14 patches
        for hidden in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if dropout > 0.0:
                encoder_layers.append(nn.Dropout(dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        in_dim = latent_dim
        for hidden in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if dropout > 0.0:
                decoder_layers.append(nn.Dropout(dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, 28 * 28))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        latent = self.encoder(features)
        reconstruction = self.decoder(latent)
        return reconstruction.reshape(x.shape[0], 1, 28, 28)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.qfilter(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


def train_hybrid_autoencoder(
    model: HybridQuanvolutionAutoencoder,
    data: torch.Tensor,
    *,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """
    Train the hybrid autoencoder. Returns a list of epoch MSE losses.
    """
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
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history


def _as_tensor(data: torch.Tensor | list[float]) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor


__all__ = [
    "QuanvolutionFilter",
    "HybridQuanvolutionAutoencoder",
    "train_hybrid_autoencoder",
]
