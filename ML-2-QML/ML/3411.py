"""HybridClassifierModel: classical neural‑network classifier with fast evaluation utilities."""
from __future__ import annotations

import torch
from torch import nn
from typing import Iterable, List, Sequence, Tuple

def build_classifier_circuit(num_features: int, depth: int, dropout: float = 0.1) -> Tuple[nn.Module, Iterable[int], List[int], List]:
    """
    Construct a configurable feed‑forward neural network.

    Parameters
    ----------
    num_features : int
        Feature dimension of the input.
    depth : int
        Number of hidden layers.
    dropout : float, optional
        Dropout probability applied after each activation.

    Returns
    -------
    network : nn.Sequential
        The constructed network.
    encoding : list[int]
        Identity mapping of input indices.
    weight_sizes : list[int]
        Number of trainable parameters per layer.
    observables : list[callable]
        Simple statistics to compute from the network output.
    """
    layers: List[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: List[int] = []

    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.extend([linear, nn.ReLU(), nn.Dropout(dropout)])
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features

    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)

    # Two observables: mean of logits and probability of class 1
    observables = [
        lambda out: out.mean(dim=-1),  # average logit
        lambda out: torch.softmax(out, dim=-1)[:, 1]  # class‑1 probability
    ]

    return network, encoding, weight_sizes, observables

class HybridClassifierModel:
    """Classical neural‑network classifier with fast batch evaluation."""
    def __init__(self, num_features: int, depth: int, dropout: float = 0.1):
        self.network, self.encoding, self.weight_sizes, self.observables = build_classifier_circuit(
            num_features, depth, dropout
        )
        self.network.eval()

    def evaluate(self, parameter_sets: Sequence[Sequence[float]]) -> List[List[float]]:
        """
        Evaluate the network for each input in `parameter_sets`.

        Parameters
        ----------
        parameter_sets : list of sequences
            Each inner sequence represents a feature vector.

        Returns
        -------
        results : list[list[float]]
            Observables computed for each input.
        """
        results: List[List[float]] = []
        with torch.no_grad():
            for params in parameter_sets:
                inputs = torch.as_tensor(params, dtype=torch.float32)
                if inputs.ndim == 1:
                    inputs = inputs.unsqueeze(0)
                outputs = self.network(inputs)
                row: List[float] = []
                for obs in self.observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        val = val.mean().item()
                    else:
                        val = float(val)
                    row.append(val)
                results.append(row)
        return results

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Return class probabilities via softmax.

        Parameters
        ----------
        inputs : torch.Tensor
            Input batch.

        Returns
        -------
        probs : torch.Tensor
            Softmax probabilities.
        """
        with torch.no_grad():
            out = self.network(inputs)
            return torch.softmax(out, dim=-1)

__all__ = ["HybridClassifierModel"]
