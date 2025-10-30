from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Tuple, Union

import numpy as np
import torch
from torch import nn

import networkx as nx

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Classical estimator that evaluates PyTorch models and provides graph utilities."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Evaluate a sequence of parameter sets on the wrapped PyTorch model.

        Parameters
        ----------
        observables
            Callables that map a model output tensor to a scalar.
        parameter_sets
            Iterable of parameter vectors that are fed to the model as a batch.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                row: List[float] = []
                for observable in observables:
                    value = observable(outputs)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)
                results.append(row)
        return results

    # ------------------------------------------------------------------
    #  Graph‑QNN utilities (classical)
    # ------------------------------------------------------------------
    @staticmethod
    def random_network(
        qnn_arch: Sequence[int], samples: int
    ) -> Tuple[List[int], List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        """Construct a random feed‑forward network and synthetic training data."""
        weights: List[torch.Tensor] = []
        for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
            weights.append(torch.randn(out_f, in_f, dtype=torch.float32))
        target_weight = weights[-1]
        training_data: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for _ in range(samples):
            features = torch.randn(target_weight.size(1), dtype=torch.float32)
            target = target_weight @ features
            training_data.append((features, target))
        return list(qnn_arch), weights, training_data, target_weight

    @staticmethod
    def feedforward(
        qnn_arch: Sequence[int],
        weights: Sequence[torch.Tensor],
        samples: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    ) -> List[List[torch.Tensor]]:
        """Forward‑propagate a list of feature vectors through the network."""
        stored: List[List[torch.Tensor]] = []
        for features, _ in samples:
            activations: List[torch.Tensor] = [features]
            current = features
            for weight in weights:
                current = torch.tanh(weight @ current)
                activations.append(current)
            stored.append(activations)
        return stored

    @staticmethod
    def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
        """Squared overlap of two normalized tensors."""
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float(torch.dot(a_norm, b_norm).item() ** 2)

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[torch.Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Build a weighted graph from pairwise fidelities."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = FastBaseEstimator.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    # ------------------------------------------------------------------
    #  Classical LSTM tagger
    # ------------------------------------------------------------------
    class LSTMTagger(nn.Module):
        """Sequence tagging model that delegates to nn.LSTM."""

        def __init__(
            self,
            embedding_dim: int,
            hidden_dim: int,
            vocab_size: int,
            tagset_size: int,
        ) -> None:
            super().__init__()
            self.hidden_dim = hidden_dim
            self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
            self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

        def forward(self, sentence: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            embeds = self.word_embeddings(sentence)
            lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
            tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
            return torch.nn.functional.log_softmax(tag_logits, dim=1)

    # ------------------------------------------------------------------
    #  Convenience: evaluate a sequence tagger
    # ------------------------------------------------------------------
    @staticmethod
    def evaluate_lstm(
        tagger: "FastBaseEstimator.LSTMTagger",
        sentence: torch.Tensor,
    ) -> torch.Tensor:
        """Run a tagger on a single sentence and return log‑probabilities."""
        return tagger(sentence)


__all__ = [
    "FastBaseEstimator",
]
