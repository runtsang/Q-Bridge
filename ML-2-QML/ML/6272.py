"""Hybrid classical graph neural network with optional self‑attention layers.

This module extends the original GraphQNN utilities by allowing each
layer to be either a standard linear transformation or a
self‑attention block.  The attention block is implemented with a
light‑weight wrapper around a NumPy‐based self‑attention routine,
mirroring the interface of the QML version so that the same
training pipeline can be reused for both classical and quantum
experiments."""
from __future__ import annotations

import itertools
import numpy as np
import torch
import networkx as nx
from typing import Iterable, List, Sequence, Tuple, Dict, Any

Tensor = torch.Tensor

# --------------------------------------------------------------------------- #
# Classical self‑attention helper
# --------------------------------------------------------------------------- #
class ClassicalSelfAttention:
    """Simple self‑attention using NumPy and PyTorch tensors."""
    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        query = torch.as_tensor(
            inputs @ rotation_params.reshape(self.embed_dim, -1),
            dtype=torch.float32,
        )
        key = torch.as_tensor(
            inputs @ entangle_params.reshape(self.embed_dim, -1),
            dtype=torch.float32,
        )
        value = torch.as_tensor(inputs, dtype=torch.float32)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ value).numpy()


# --------------------------------------------------------------------------- #
# Core utilities
# --------------------------------------------------------------------------- #
def _random_linear(in_features: int, out_features: int) -> Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(
    qnn_arch: Sequence[int],
    attention_cfg: Sequence[Dict[str, Any] | None],
    samples: int,
) -> Tuple[Sequence[int], List[Tensor | Dict[str, Any]], List[Tuple[Tensor, Tensor]], Tensor]:
    """
    Build a hybrid architecture.

    ``attention_cfg`` is a list of the same length as ``qnn_arch``; ``None``
    indicates a standard linear layer, while a dictionary describes an
    attention block, e.g. ``{'type':'attention','embed_dim':4}``.
    """
    layers: List[Tensor | Dict[str, Any]] = []
    for layer, cfg in zip(qnn_arch[:-1], attention_cfg):
        if cfg is None:
            layers.append(_random_linear(layer, qnn_arch[layer + 1]))
        else:
            assert cfg["type"] == "attention"
            attn = ClassicalSelfAttention(cfg["embed_dim"])
            rot = np.random.normal(size=3 * cfg["embed_dim"])
            ent = np.random.normal(size=cfg["embed_dim"] - 1)
            layers.append(
                {
                    "attention": attn,
                    "rot": rot,
                    "ent": ent,
                }
            )
    target_weight = _random_linear(qnn_arch[-2], qnn_arch[-1])
    training_data = random_training_data(target_weight, samples)
    return qnn_arch, layers, training_data, target_weight


def feedforward(
    qnn_arch: Sequence[int],
    layers: Sequence[Tensor | Dict[str, Any]],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    stored: List[List[Tensor]] = []
    for features, _ in samples:
        activations = [features]
        current = features
        for layer in layers:
            if isinstance(layer, dict):
                # attention layer
                attn = layer["attention"]
                out = attn.run(layer["rot"], layer["ent"], current.numpy())
                current = torch.as_tensor(out, dtype=torch.float32)
            else:
                current = torch.tanh(layer @ current)
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
    secondary: float | None = None,
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


__all__ = [
    "ClassicalSelfAttention",
    "random_network",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
    "random_training_data",
]
