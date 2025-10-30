import itertools
from typing import Iterable, List, Sequence, Tuple, Optional

import networkx as nx
import torch
import torch.nn.functional as F

Tensor = torch.Tensor

def _random_linear(in_features: int, out_features: int) -> Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32, requires_grad=True)

def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
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
    stored: List[List[Tensor]] = []
    for features, _ in samples:
        activations = [features]
        current = features
        for weight in weights:
            current = torch.tanh(weight @ current)
            activations.append(current)
        stored.append(activations)
    return stored

def state_fidelity(a: Tensor, b: Tensor) -> float:
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
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

class GraphQNN:
    """
    A hybrid graph neural network that can operate on classical data
    or be paired with a quantum backend.  The class retains the original
    feed‑forward and fidelity utilities while adding a hybrid loss and
    a lightweight training loop.  The weights are shared with a
    possible quantum implementation, enabling a seamless transition
    between classical and quantum training.
    """

    def __init__(
        self,
        arch: Sequence[int],
        device: Optional[torch.device] = None,
        *,
        loss_weight: float = 1.0,
        early_stopping: Optional[bool] = None,
        checkpoint_path: Optional[str] = None,
    ):
        self.arch = list(arch)
        self.device = device or torch.device("cpu")
        self.loss_weight = loss_weight
        self.early_stopping = early_stopping
        self.checkpoint_path = checkpoint_path

        # Randomly initialise weights (shared with quantum backend)
        self.weights: List[Tensor] = [
            _random_linear(in_f, out_f).to(self.device) for in_f, out_f in zip(self.arch[:-1], self.arch[1:])
        ]

        # Graph for fidelity‑based adjacency
        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(len(self.arch)))

    def feedforward(self, samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
        """Return all intermediate activations for the provided samples."""
        return feedforward(self.arch, self.weights, samples)

    def fidelity_adjacency(
        self,
        states: Sequence[Tensor],
        threshold: float,
        *,
        secondary: Optional[float] = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        return fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)

    def hybrid_loss(
        self,
        outputs: Tensor,
        targets: Tensor,
        state_outputs: Optional[Tensor] = None,
        state_targets: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute a weighted sum of MSE between outputs and targets and a
        fidelity term between quantum state outputs and targets (if supplied).
        """
        mse = F.mse_loss(outputs, targets)
        if state_outputs is not None and state_targets is not None:
            # Fidelity term: 1 - average fidelity
            fid_sum = sum(state_fidelity(out, tgt) for out, tgt in zip(state_outputs, state_targets))
            fidelity_loss = 1.0 - fid_sum / len(state_outputs)
            return mse + self.loss_weight * torch.tensor(fidelity_loss, dtype=torch.float32, device=self.device)
        return mse

    def train(
        self,
        dataset: Iterable[Tuple[Tensor, Tensor]],
        epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 32,
        verbose: bool = True,
    ) -> List[float]:
        """Simple training loop using Adam."""
        optimizer = torch.optim.Adam(self.weights, lr=lr)
        loss_history: List[float] = []

        data = list(dataset)
        for epoch in range(epochs):
            epoch_loss = 0.0
            torch.random.manual_seed(epoch)
            torch.randperm(len(data))

            for i in range(0, len(data), batch_size):
                batch = data[i : i + batch_size]
                features = torch.stack([x[0] for x in batch]).to(self.device)
                targets = torch.stack([x[1] for x in batch]).to(self.device)

                optimizer.zero_grad()
                outputs = self._forward(features)
                loss = self.hybrid_loss(outputs, targets)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * len(batch)

            epoch_loss /= len(data)
            loss_history.append(epoch_loss)

            if verbose:
                print(f"Epoch {epoch+1}/{epochs} – loss: {epoch_loss:.6f}")

            if self.early_stopping and epoch > 0 and loss_history[-1] > loss_history[-2]:
                if verbose:
                    print("Early stopping triggered.")
                break

            if self.checkpoint_path:
                torch.save(
                    {"arch": self.arch, "weights": [w.cpu() for w in self.weights]},
                    self.checkpoint_path,
                )

        return loss_history

    def _forward(self, batch: Tensor) -> Tensor:
        """Return the final layer activations for a batch of inputs."""
        current = batch
        for weight in self.weights:
            current = torch.tanh(weight @ current.T).T
        return current

    def export_surrogate(self) -> List[Tensor]:
        """Return a list of frozen weights that can be used as a pure classical model."""
        return [w.detach().cpu() for w in self.weights]

    @staticmethod
    def random_network(qnn_arch: Sequence[int], samples: int):
        return random_network(qnn_arch, samples)

    @staticmethod
    def random_training_data(weight: Tensor, samples: int):
        return random_training_data(weight, samples)

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        return state_fidelity(a, b)

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[Tensor],
        threshold: float,
        *,
        secondary: Optional[float] = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        return fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)

__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "GraphQNN",
]
