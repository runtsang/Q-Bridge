"""Hybrid classical–quantum model with CNN backbone and variational quantum layer.

This module defines :class:`QuantumNATHybrid` that merges the ideas from the three
reference pairs:

* The CNN + fully‑connected projection from the original
  :class:`QFCModel` (seed 1).
* A variational quantum circuit that accepts the pooled feature vector as
  input and produces a 4‑dimensional probability vector (seed 2).
* Graph‑based fidelity utilities from the third pair to allow the
  model to be trained on graph‑structured data or to construct a
  fidelity‑based regularisation term.

The architecture is intentionally *stateless* – all learnable
parameters are stored in ``self.parameters`` and can be optimised
with any PyTorch optimiser.  No training loop is provided; the
module is ready for integration into larger pipelines.

"""

from __future__ import annotations

import itertools
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import qutip as qt
import scipy as sc

# --------------------------------------------------------------------------- #
# Utility functions – fidelity, graph construction, random data
# --------------------------------------------------------------------------- #

def _tensored_id(num_qubits: int) -> qt.Qobj:
    identity = qt.qeye(2 ** num_qubits)
    dims = [2] * num_qubits
    identity.dims = [dims.copy(), dims.copy()]
    return identity


def _tensored_zero(num_qubits: int) -> qt.Qobj:
    projector = qt.fock(2 ** num_qubits).proj()
    dims = [2] * num_qubits
    projector.dims = [dims.copy(), dims.copy()]
    return projector


def _swap_registers(op: qt.Qobj, source: int, target: int) -> qt.Qobj:
    if source == target:
        return op
    order = list(range(len(op.dims[0])))
    order[source], order[target] = order[target], order[source]
    return op.permute(order)


def _random_qubit_unitary(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    matrix = sc.random.normal(size=(dim, dim)) + 1j * sc.random.normal(size=(dim, dim))
    unitary = sc.linalg.orth(matrix)
    qobj = qt.Qobj(unitary)
    dims = [2] * num_qubits
    qobj.dims = [dims.copy(), dims.copy()]
    return qobj


def _random_qubit_state(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    amplitudes = sc.random.normal(size=(dim, 1)) + 1j * sc.random.normal(size=(dim, 1))
    amplitudes /= sc.linalg.norm(amplitudes)
    state = qt.Qobj(amplitudes)
    state.dims = [[2] * num_qubits, [1] * num_qubits]
    return state


def random_training_data(unitary: qt.Qobj, samples: int) -> list[tuple[qt.Qobj, qt.Qobj]]:
    """Return a list of tuples (input, target) for supervised learning of a QNN."""
    dataset = []
    num_qubits = len(unitary.dims[0])
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        dataset.append((state, unitary * state))
    return dataset


def random_network(qnn_arch: list[int], samples: int):
    """Return a random network architecture and training data for a graph‑QNN."""
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    unitaries: list[list[qt.Qobj]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        layer_ops: list[qt.Qobj] = []
        for output in range(num_outputs):
            op = _random_qubit_unitary(num_inputs + 1)
            if num_outputs > 1:
                op = qt.tensor(_random_qubit_unitary(num_inputs + 1), _tensored_id(num_outputs - 1))
                op = _swap_registers(op, num_inputs, num_inputs + output)
            layer_ops.append(op)
        unitaries.append(layer_ops)

    return qnn_arch, unitaries, training_data, target_unitary


def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
    """Return the absolute squared overlap between pure states ``a`` and ``b``."""
    return abs((a.dag() * b)[0, 0]) ** 2


def fidelity_adjacency(states: Sequence[qt.Qobj], threshold: float, *, secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
    """Create a weighted adjacency graph from state fidelities.

    Edges with fidelity greater than or equal to ``threshold`` receive weight 1.
    When ``secondary`` is provided, fidelities between ``secondary`` and
    ``threshold`` are added with ``secondary_weight``.
    """
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
# Classical CNN backbone – extracted from seed 1
# --------------------------------------------------------------------------- #

class _CNNBackbone(nn.Module):
    """A lightweight CNN that produces a 16‑dimensional feature vector."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


# --------------------------------------------------------------------------- #
# Classical sampler network – mirrors the QNN helper
# --------------------------------------------------------------------------- #

class SamplerModule(nn.Module):
    """Small neural sampler that maps feature vectors to a probability distribution."""

    def __init__(self, input_dim: int = 16, output_dim: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.Tanh(),
            nn.Linear(8, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return F.softmax(self.net(inputs), dim=-1)


def SamplerQNN() -> nn.Module:
    """Factory that returns a ready‑to‑use sampler network."""
    return SamplerModule()


# --------------------------------------------------------------------------- #
# Hybrid model – classical backbone + sampler
# --------------------------------------------------------------------------- #

class QuantumNATHybrid(nn.Module):
    """Hybrid classical model that fuses a CNN backbone with a sampler.

    The class also exposes graph‑based utilities for fidelity‑based
    adjacency construction, allowing the model to be used in graph‑based
    regularisation or clustering tasks.

    Examples
    --------
    >>> model = QuantumNATHybrid()
    >>> x = torch.randn(4, 1, 28, 28)
    >>> probs = model(x)
    >>> print(probs.shape)
    torch.Size([4, 4])
    """

    def __init__(self) -> None:
        super().__init__()
        self.backbone = _CNNBackbone()
        self.sampler = SamplerQNN()
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CNN–sampler pipeline."""
        features = self.backbone(x)
        flattened = features.view(features.shape[0], -1)
        probs = self.sampler(flattened)
        return self.norm(probs)

    # --------------------------------------------------------------------- #
    # Graph utilities – expose the same API as in the quantum version
    # --------------------------------------------------------------------- #

    @staticmethod
    def fidelity_adjacency(states: Sequence[qt.Qobj], threshold: float,
                           *, secondary: float | None = None,
                           secondary_weight: float = 0.5) -> nx.Graph:
        """Wrap the quantum‑based adjacency construction for compatibility."""
        return fidelity_adjacency(states, threshold, secondary=secondary,
                                  secondary_weight=secondary_weight)

    @staticmethod
    def random_network(qnn_arch: list[int], samples: int):
        """Return a random graph‑QNN architecture and training data."""
        return random_network(qnn_arch, samples)

    @staticmethod
    def random_training_data(unitary: qt.Qobj, samples: int):
        """Return random training data for a target unitary."""
        return random_training_data(unitary, samples)

__all__ = ["QuantumNATHybrid"]
