import torch
from torch import nn
from dataclasses import dataclass
from typing import Tuple, List, Iterable, Dict
import itertools
import networkx as nx

try:
    import qutip as qt  # optional, used only for fidelity graphs
except ImportError:  # pragma: no cover
    qt = None


@dataclass
class AutoencoderConfig:
    """Configuration for :class:`HybridAutoencoder`."""

    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1
    use_quantum: bool = False  # flag for quantum submodules (currently unused in ML)
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 50
    batch_size: int = 64
    device: torch.device | None = None


def _build_transformer(
    in_dim: int, hidden_dims: Tuple[int,...], out_dim: int
) -> nn.Sequential:
    """Simple linear‑ReLU transformer block."""
    layers: List[nn.Module] = []
    prev = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.1))
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class HybridAutoencoder(nn.Module):
    """
    Classical autoencoder that can optionally be extended with quantum layers.
    The API mirrors that of the original AutoencoderNet but adds a
    ``fidelity_graph`` helper that builds a graph of latent fidelities using
    :mod:`qutip`.
    """

    def __init__(self, cfg: Dict | AutoencoderConfig) -> None:
        super().__init__()
        if isinstance(cfg, dict):
            cfg = AutoencoderConfig(**cfg)
        self.cfg = cfg
        self.device = cfg.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Encoder / Decoder
        self.encoder = _build_transformer(
            cfg.input_dim, cfg.hidden_dims, cfg.latent_dim
        ).to(self.device)
        self.decoder = _build_transformer(
            cfg.latent_dim, cfg.hidden_dims[::-1], cfg.input_dim
        ).to(self.device)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(inputs))

    def train_autoencoder(
        self, data: torch.Tensor, *, epochs: int | None = None
    ) -> List[float]:
        """Train the autoencoder and return a list of epoch losses."""
        epochs = epochs or self.cfg.epochs
        data = data.to(self.device)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(data),
            batch_size=self.cfg.batch_size,
            shuffle=True,
        )
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay
        )
        loss_fn = nn.MSELoss()
        history: List[float] = []

        for _ in range(epochs):
            epoch_loss = 0.0
            for batch, in loader:
                batch = batch.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                recon = self(batch)
                loss = loss_fn(recon, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch.size(0)
            epoch_loss /= len(loader.dataset)
            history.append(epoch_loss)
        return history

    def fidelity_graph(self, latent_samples: torch.Tensor, threshold: float = 0.8
    ) -> nx.Graph:
        """
        Build a weighted adjacency graph from the latent space using
        quantum state fidelity.  The function requires :mod:`qutip`; if it
        is not available a RuntimeError is raised.
        """
        if qt is None:  # pragma: no cover
            raise RuntimeError("qutip is required for fidelity graph construction")
        latent_samples = latent_samples.cpu().detach()
        states = [qt.Qobj(state.numpy()) for state in latent_samples]
        return fidelity_adjacency(states, threshold)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<HybridAutoencoder latent_dim={self.cfg.latent_dim}>"


__all__ = ["HybridAutoencoder", "AutoencoderConfig"]
