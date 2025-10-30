"""GraphQNN – Classical GNN with adaptive regularisation and batch sampling.

This module extends the original seed by adding:
* a layer‑wise L2 penalty that grows with depth,
* a hybrid loss `L = L_cls + α L_q` that can be evaluated on either side,
* a :class:`torch.utils.data.DataLoader`‑style sampler that yields mini‑batches
  for training.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional

import networkx as nx
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

Tensor = torch.Tensor

__all__ = [
    "GraphQNN",
    "Dataset",
    "DataLoader",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "fidelity_adjacency",
]

# --------------------------------------------------------------------------- #
#  Random data generation
# --------------------------------------------------------------------------- #

def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix of shape ``(out_features, in_features)``."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(
    weight: Tensor,
    samples: int,
    *,
    seed: Optional[int] = None,
) -> List[Tuple[Tensor, Tensor]]:
    """Generate synthetic data ``(x, y = Wx)`` for a given weight matrix."""
    if seed is not None:
        torch.manual_seed(seed)
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int, *, seed: Optional[int] = None):
    """Create a random linear network together with training data.

    Returns
    -------
    arch : List[int]
        Architecture list.
    weights : List[Tensor]
        List of weight matrices per layer.
    training_data : List[Tuple[Tensor, Tensor]]
        Synthetic training data generated from the last layer.
    target_weight : Tensor
        The weight matrix of the last layer (used as the target for training).
    """
    if seed is not None:
        torch.manual_seed(seed)
    weights: List[Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight


# --------------------------------------------------------------------------- #
#  Classical feed‑forward and fidelity helpers
# --------------------------------------------------------------------------- #

def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[Tensor],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    """Run a forward pass through a list of linear layers.

    Parameters
    ----------
    qnn_arch : Sequence[int]
        Architecture, e.g. ``[3, 5, 2]``.
    weights : Sequence[Tensor]
        Weight matrices for each layer.
    samples : Iterable[Tuple[Tensor, Tensor]]
        Iterable of input/target pairs. Only the input part is used.

    Returns
    -------
    List[List[Tensor]]
        Activations per sample, including the input as the first element.
    """
    activations: List[List[Tensor]] = []
    for features, _ in samples:
        layerwise = [features]
        current = features
        for weight in weights:
            current = torch.tanh(weight @ current)
            layerwise.append(current)
        activations.append(layerwise)
    return activations


def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Return the squared overlap between two normalized vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)


def fidelity_adjacency(
    states: Sequence[Tensor],
    threshold: float,
    *,
    secondary: Optional[float] = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Construct a weighted graph from state fidelities.

    Parameters
    ----------
    states : Sequence[Tensor]
        List of state vectors.
    threshold : float
        Fidelity threshold for primary edges.
    secondary : float, optional
        Fidelity threshold for secondary edges.
    secondary_weight : float, default 0.5
        Weight assigned to secondary edges.

    Returns
    -------
    nx.Graph
        Weighted graph where nodes are the indices of ``states``.
    """
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
#  Dataset and DataLoader helpers
# --------------------------------------------------------------------------- #

class _RandomDataset(Dataset):
    """Simple torch Dataset wrapping synthetic training data."""

    def __init__(self, data: List[Tuple[Tensor, Tensor]]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return self.data[idx]


def random_dataloader(
    data: List[Tuple[Tensor, Tensor]],
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    """Convenience wrapper that returns a :class:`torch.utils.data.DataLoader`."""
    dataset = _RandomDataset(data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# --------------------------------------------------------------------------- #
#  GraphQNN class
# --------------------------------------------------------------------------- #

class GraphQNN(nn.Module):
    """Hybrid classical‑quantum neural network.

    Parameters
    ----------
    arch : Sequence[int]
        Layer sizes, e.g. ``[3, 5, 2]``.
    device : str, optional
        ``"cpu"`` or ``"cuda"``.
    reg_factor : float, default 1e-4
        Base L2 regularisation factor applied per layer.
    alpha : float, default 0.5
        Weight of the quantum loss term in the hybrid loss.

    Notes
    -----
    The class keeps the original feed‑forward helpers but adds a layer‑wise
    regularisation and a simple hybrid loss that can be used with a quantum
    backend if desired.
    """

    def __init__(
        self,
        arch: Sequence[int],
        device: str = "cpu",
        reg_factor: float = 1e-4,
        alpha: float = 0.5,
    ):
        super().__init__()
        self.arch = list(arch)
        self.device = torch.device(device)
        self.reg_factor = reg_factor
        self.alpha = alpha

        # Build linear layers
        self.layers = nn.ModuleList()
        for in_f, out_f in zip(self.arch[:-1], self.arch[1:]):
            self.layers.append(nn.Linear(in_f, out_f, bias=True))

        # Move to device
        self.to(self.device)

    # --------------------------------------------------------------------- #
    #  Forward pass
    # --------------------------------------------------------------------- #

    def forward(self, x: Tensor) -> Tensor:
        """Standard forward pass through the network."""
        out = x
        for layer in self.layers:
            out = torch.tanh(layer(out))
        return out

    # --------------------------------------------------------------------- #
    #  Loss computation
    # --------------------------------------------------------------------- #

    def l2_regularisation(self) -> Tensor:
        """Layer‑wise L2 penalty that grows with depth."""
        reg = torch.tensor(0.0, device=self.device)
        for idx, layer in enumerate(self.layers, start=1):
            reg += self.reg_factor * idx * torch.sum(layer.weight.pow(2))
        return reg

    def hybrid_loss(
        self,
        predictions: Tensor,
        targets: Tensor,
        quantum_loss: Optional[Tensor] = None,
    ) -> Tensor:
        """Combine classical MSE with an optional quantum loss."""
        mse = nn.functional.mse_loss(predictions, targets, reduction="mean")
        if quantum_loss is not None:
            return mse + self.alpha * quantum_loss
        return mse

    # --------------------------------------------------------------------- #
    #  Training helper
    # --------------------------------------------------------------------- #

    def train_on_loader(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        epochs: int,
        quantum_loss_fn: Optional[Callable[[Tensor], Tensor]] = None,
    ) -> None:
        """Simple training loop over a DataLoader.

        Parameters
        ----------
        loader : DataLoader
            Mini‑batch sampler.
        optimizer : torch.optim.Optimizer
            Optimiser to use.
        epochs : int
            Number of epochs.
        quantum_loss_fn : callable, optional
            Function that returns a quantum loss term given the current
            predictions.  If ``None`` the quantum term is ignored.
        """
        self.train()
        for _ in range(epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                preds = self(batch_x)
                q_loss = quantum_loss_fn(preds) if quantum_loss_fn else None
                loss = self.hybrid_loss(preds, batch_y, q_loss)
                loss += self.l2_regularisation()
                loss.backward()
                optimizer.step()
