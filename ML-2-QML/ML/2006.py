from __future__ import annotations

import itertools
import math
from typing import Iterable, Sequence, List, Tuple, Optional

import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim

Tensor = torch.Tensor


def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a weight matrix initialized with a standard normal distribution."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate input–target pairs for a linear mapping given a target weight matrix."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.shape[1], dtype=torch.float32)
        target = weight @ features
        # Add a tiny Gaussian perturbation to emulate realistic noise
        target += 0.01 * torch.randn_like(target)
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a random classical feed‑forward network and a small training set."""
    weights: List[Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight


def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[Tensor],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    """Run the network on an iterable of input–target tuples and collect activations."""
    stored: List[List[Tensor]] = []
    for features, _ in samples:
        activations: List[Tensor] = [features]
        current = features
        for weight in weights:
            current = torch.tanh(weight @ current)
            activations.append(current)
        stored.append(activations)
    return stored


def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Absolute squared overlap between two state vectors."""
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
    """Build a weighted graph from pairwise state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


class GraphQNN__gen206(nn.Module):
    """
    Classical Graph‑based QNN with optional L2 regularisation and hybrid fidelity loss.
    """

    def __init__(self, arch: Sequence[int], reg_coeff: float = 0.0):
        super().__init__()
        self.arch = list(arch)
        self.reg_coeff = reg_coeff

        layers = []
        for in_f, out_f in zip(arch[:-1], arch[1:]):
            layers.append(nn.Linear(in_f, out_f))
        self.layers = nn.ModuleList(layers)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = torch.tanh(layer(x))
        return x

    # ------------------------------------------------------------------
    # Losses
    # ------------------------------------------------------------------
    def loss_mse(self, pred: Tensor, target: Tensor) -> Tensor:
        return nn.functional.mse_loss(pred, target)

    def loss_fidelity(self, pred: Tensor, target: Tensor) -> Tensor:
        # Treat predictions and targets as state vectors
        pred_norm = pred / (torch.norm(pred) + 1e-12)
        target_norm = target / (torch.norm(target) + 1e-12)
        return 1.0 - torch.dot(pred_norm, target_norm).pow(2)

    def loss_hybrid(self, pred: Tensor, target: Tensor) -> Tensor:
        return self.loss_mse(pred, target) + self.loss_fidelity(pred, target)

    # ------------------------------------------------------------------
    # Regularisation
    # ------------------------------------------------------------------
    def regularization(self) -> Tensor:
        reg = torch.tensor(0.0, device=self.layers[0].weight.device)
        for layer in self.layers:
            reg += torch.norm(layer.weight, p=2)
        return self.reg_coeff * reg

    # ------------------------------------------------------------------
    # Training helper
    # ------------------------------------------------------------------
    def train_on_dataset(
        self,
        dataset: List[Tuple[Tensor, Tensor]],
        epochs: int = 100,
        lr: float = 1e-3,
        early_stop_patience: int = 10,
        loss_fn: str = "mse",
    ) -> List[float]:
        """
        Train the network on the supplied dataset.

        Parameters
        ----------
        dataset : list of (input, target) pairs
        epochs : maximum number of epochs
        lr : learning rate for Adam optimiser
        early_stop_patience : number of epochs with no improvement before stopping
        loss_fn : choice of'mse', 'fidelity', or 'hybrid'

        Returns
        -------
        history : list of epoch‑wise loss values
        """
        optimizer = optim.Adam(self.parameters(), lr=lr)
        best_loss = math.inf
        best_state_dict = None
        patience_counter = 0
        history: List[float] = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            for x, y in dataset:
                optimizer.zero_grad()
                y_pred = self.forward(x)
                if loss_fn == "mse":
                    loss = self.loss_mse(y_pred, y)
                elif loss_fn == "fidelity":
                    loss = self.loss_fidelity(y_pred, y)
                else:  # hybrid
                    loss = self.loss_hybrid(y_pred, y)
                loss += self.regularization()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            epoch_loss /= len(dataset)
            history.append(epoch_loss)

            # early‑stopping logic
            if epoch_loss < best_loss - 1e-6:
                best_loss = epoch_loss
                best_state_dict = {k: v.clone() for k, v in self.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    break

        if best_state_dict is not None:
            self.load_state_dict(best_state_dict)

        return history


__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "GraphQNN__gen206",
]
