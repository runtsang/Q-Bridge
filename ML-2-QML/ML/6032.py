"""Hybrid classical autoencoder module.

Provides a configurable MLP encoder/decoder, an optional quantum refinement step,
and utilities for building a fidelity‑based graph of latent embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import networkx as nx
import itertools

# --------------------------------------------------------------------------- #
#  Utility: tensor conversion --------------------------------------------

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Return a float32 tensor on the current default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

# --------------------------------------------------------------------------- #
#  Graph utilities (adapted from GraphQNN seed) -------------------------

def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Return the squared cosine similarity between two vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)

def fidelity_adjacency(states: Sequence[torch.Tensor], threshold: float,
                       *, secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
    """Build a weighted graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

# --------------------------------------------------------------------------- #
#  Configuration ---------------------------------------------------------

@dataclass
class QuantumAutoencoderHybridConfig:
    """Configuration for the hybrid autoencoder."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    # Quantum parameters
    num_trash: int = 2
    ansatz_reps: int = 5
    swap_test: bool = True
    # Training hyper‑parameters
    lr: float = 1e-3
    epochs: int = 100
    batch_size: int = 64
    device: Optional[torch.device] = None

# --------------------------------------------------------------------------- #
#  Classical encoder/decoder --------------------------------------------

class _MLPEncoder(nn.Module):
    """Fully‑connected encoder."""
    def __init__(self, cfg: QuantumAutoencoderHybridConfig) -> None:
        super().__init__()
        layers = []
        in_dim = cfg.input_dim
        for hidden in cfg.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class _MLPDecoder(nn.Module):
    """Fully‑connected decoder that concatenates classical and quantum latent."""
    def __init__(self, cfg: QuantumAutoencoderHybridConfig) -> None:
        super().__init__()
        layers = []
        in_dim = cfg.latent_dim * 2   # we will concatenate classical and quantum latent
        for hidden in cfg.hidden_dims[::-1]:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# --------------------------------------------------------------------------- #
#  Hybrid autoencoder ----------------------------------------------------

class QuantumAutoencoderHybrid(nn.Module):
    """Hybrid autoencoder that optionally refines the latent representation with a quantum circuit."""
    def __init__(self, cfg: QuantumAutoencoderHybridConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = _MLPEncoder(cfg)
        self.decoder = _MLPDecoder(cfg)
        # The quantum part is optional and will be added by the QML module.
        self.quantum_circuit = None
        self.qnn = None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass: encoder → (quantum refine) → decoder."""
        z = self.encode(x)
        # If quantum refinement is available, refine the latent.
        if self.qnn is not None:
            z = self.refine_quantum(z)
        return self.decode(z)

    def refine_quantum(self, z: torch.Tensor) -> torch.Tensor:
        """Refine the latent vector using the quantum circuit (via SamplerQNN)."""
        if self.qnn is None:
            return z
        # The quantum circuit expects a real‑valued input vector; we feed the latent directly.
        # The output of the SamplerQNN is a probability distribution; we use the first
        # probability as a scalar refinement factor.
        probs = self.qnn.forward(z)
        # Use the first probability as a scaling factor (ensures positivity).
        scale = probs[:, 0].unsqueeze(-1)
        return z * scale

    def compute_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction loss."""
        z = self.encode(x)
        recon = self.decode(z)
        loss = F.mse_loss(recon, x, reduction="sum")
        return loss

# --------------------------------------------------------------------------- #
#  Training helper -------------------------------------------------------

def train_hybrid_autoencoder(model: QuantumAutoencoderHybrid,
                             data: torch.Tensor,
                             *,
                             epochs: int | None = None,
                             batch_size: int | None = None,
                             lr: float | None = None,
                             weight_decay: float = 0.0) -> List[float]:
    """Train the hybrid autoencoder and return the loss history."""
    cfg = model.cfg
    device = cfg.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size or cfg.batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr or cfg.lr,
                                 weight_decay=weight_decay)
    history: List[float] = []
    for epoch in range(epochs or cfg.epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = model.compute_loss(batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

# --------------------------------------------------------------------------- #
#  Fidelity graph construction -------------------------------------------

def fidelity_graph(latent_vectors: torch.Tensor,
                   threshold: float,
                   *,
                   secondary: float | None = None,
                   secondary_weight: float = 0.5) -> nx.Graph:
    """Build a weighted adjacency graph of latent vectors based on state fidelities."""
    states = [vec.detach().cpu() for vec in latent_vectors]
    return fidelity_adjacency(states, threshold,
                               secondary=secondary,
                               secondary_weight=secondary_weight)

__all__ = [
    "QuantumAutoencoderHybrid",
    "QuantumAutoencoderHybridConfig",
    "train_hybrid_autoencoder",
    "fidelity_graph",
]
