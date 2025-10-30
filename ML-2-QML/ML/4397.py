"""Hybrid classical autoencoder that fuses convolution, sampler, and graph‑regularized latent space."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, Sequence, List, Optional
import itertools

import torch
from torch import nn
import torch.nn.functional as F
import networkx as nx
from torch.utils.data import DataLoader, TensorDataset

# --------------------------------------------------------------------------- #
# 1. Helper utilities
# --------------------------------------------------------------------------- #
def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Return a float32 tensor on the current default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

class ConvFilter(nn.Module):
    """2‑D convolutional filter with a sigmoid threshold."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.view(activations.size(0), -1).mean(dim=1, keepdim=True)

class SamplerModule(nn.Module):
    """Tiny neural sampler producing a 2‑D probability distribution."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(x), dim=-1)

# --------------------------------------------------------------------------- #
# 2. Graph utilities (adapted from GraphQNN)
# --------------------------------------------------------------------------- #
def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Squared overlap between two unit‑normed vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float((a_norm @ b_norm).item() ** 2)

def fidelity_adjacency(
    states: Sequence[torch.Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Build a weighted graph where edges encode state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(a, b)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

# --------------------------------------------------------------------------- #
# 3. Configuration
# --------------------------------------------------------------------------- #
@dataclass
class HybridAutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    use_conv: bool = False
    kernel_size: int = 2
    conv_threshold: float = 0.0
    use_sampler: bool = False
    graph_fidelity_threshold: float = 0.9
    graph_secondary_threshold: float | None = None

# --------------------------------------------------------------------------- #
# 4. Hybrid autoencoder
# --------------------------------------------------------------------------- #
class HybridAutoencoderNet(nn.Module):
    """Classical autoencoder that optionally uses convolution, a sampler, and graph regularisation."""
    def __init__(self, config: HybridAutoencoderConfig) -> None:
        super().__init__()
        self.config = config

        # Encoder
        encoder_layers: List[nn.Module] = []
        in_dim = config.input_dim
        if config.use_conv:
            self.conv = ConvFilter(kernel_size=config.kernel_size, threshold=config.conv_threshold)
            in_dim = 1  # after conv we collapse to a scalar per sample
        else:
            self.conv = None
        for hidden in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Sampler
        self.sampler = SamplerModule() if config.use_sampler else None

        # Decoder
        decoder_layers: List[nn.Module] = []
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
        if self.conv is not None:
            # Expect inputs of shape (batch, 1, H, W)
            inputs = self.conv(inputs)
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(inputs))

# --------------------------------------------------------------------------- #
# 5. Factory
# --------------------------------------------------------------------------- #
def HybridAutoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    use_conv: bool = False,
    kernel_size: int = 2,
    conv_threshold: float = 0.0,
    use_sampler: bool = False,
    graph_fidelity_threshold: float = 0.9,
    graph_secondary_threshold: float | None = None,
) -> HybridAutoencoderNet:
    config = HybridAutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        use_conv=use_conv,
        kernel_size=kernel_size,
        conv_threshold=conv_threshold,
        use_sampler=use_sampler,
        graph_fidelity_threshold=graph_fidelity_threshold,
        graph_secondary_threshold=graph_secondary_threshold,
    )
    return HybridAutoencoderNet(config)

# --------------------------------------------------------------------------- #
# 6. Training routine
# --------------------------------------------------------------------------- #
def train_hybrid_autoencoder(
    model: HybridAutoencoderNet,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
    graph_regularization: bool = True,
) -> list[float]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: list[float] = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction = model(batch)
            loss = loss_fn(reconstruction, batch)

            if graph_regularization and model.config.graph_fidelity_threshold is not None:
                z = model.encode(batch).detach()
                # pairwise fidelities
                norms = z.norm(dim=1, keepdim=True)
                z_norm = z / (norms + 1e-12)
                fids = (z_norm @ z_norm.t()).clamp(0, 1)
                mask = torch.triu((fids >= model.config.graph_fidelity_threshold).float(), diagonal=1)
                diff = (z.unsqueeze(1) - z.unsqueeze(0)).pow(2).sum(-1)
                penalty = torch.sum(diff * mask)
                loss += 0.01 * penalty

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

__all__ = [
    "HybridAutoencoder",
    "HybridAutoencoderNet",
    "HybridAutoencoderConfig",
    "train_hybrid_autoencoder",
]
