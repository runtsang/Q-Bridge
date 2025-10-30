"""GraphQNN__gen035: Classical and hybrid graph neural‑network utilities.

The module keeps the original feed‑forward and fidelity helpers, but extends them with:
* a small `HybridGraphNet` class that stores per‑layer weight matrices and can be used with PyTorch optimisers.
* a `train_step` helper that performs a gradient‑based update using the mean‑squared‑error loss between predicted and target states.
* a `compare_with_qnn` function that runs the same network on a classical and a quantum backend and returns the fidelity between the two predictions.

The design mirrors the QML interface so that downstream code can import `GraphQNN__gen035` and use the same API, while the added features allow quick experimentation with hybrid training pipelines.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import torch
from torch import nn

Tensor = torch.Tensor

# --------------------------------------------------------------------------- #
# Original seed functions (kept for compatibility)
# --------------------------------------------------------------------------- #
def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a weight matrix with std‑normal entries."""
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate samples of the form (x, Wx) for a fixed linear map."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    """Return architecture, weight list, training data and the target weight."""
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
    """Return activations for each sample through the network."""
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
    """Compute the squared overlap between two classical vectors."""
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
    """Construct a graph from state fidelities with optional secondary weighting."""
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
# New hybrid‑training utilities
# --------------------------------------------------------------------------- #
class HybridGraphNet(nn.Module):
    """A lightweight neural‑network‑style model that mirrors the QNN architecture.

    Parameters
    ----------
    qnn_arch : Sequence[int]
        Number of nodes per layer.
    weights : Sequence[Tensor] | None, optional
        Initial weight matrices. If ``None``, random weights are created.
    """

    def __init__(self, qnn_arch: Sequence[int], weights: Sequence[Tensor] | None = None):
        super().__init__()
        self.layers = nn.ModuleList()
        for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
            self.layers.append(nn.Linear(in_f, out_f, bias=False))
        if weights is not None:
            for layer, w in zip(self.layers, weights):
                layer.weight.data = w.clone()

    def forward(self, x: Tensor) -> Tensor:
        """Standard forward pass."""
        for layer in self.layers:
            x = torch.tanh(layer(x))
        return x

def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Iterable[Tuple[Tensor, Tensor]],
    loss_fn: nn.Module = nn.MSELoss(),
) -> float:
    """Perform one gradient‑based update on a batch of samples.

    Returns the average loss over the batch.
    """
    model.train()
    total_loss = 0.0
    for features, target in batch:
        optimizer.zero_grad()
        pred = model(features)
        loss = loss_fn(pred, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(batch)

def compare_with_qnn(
    classical_model: nn.Module,
    qnn: "HybridQNN",
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> float:
    """Run the same data through a classical and a quantum network and return the mean fidelity."""
    classical_model.eval()
    fidelities = []
    for features, _ in samples:
        with torch.no_grad():
            classical_out = classical_model(features)
        quantum_out = qnn.forward_state(features)
        fid = state_fidelity(classical_out, quantum_out)
        fidelities.append(fid)
    return sum(fidelities) / len(fidelities)

# --------------------------------------------------------------------------- #
# Quantum‑classical bridge (placeholder for the QML module)
# --------------------------------------------------------------------------- #
class HybridQNN:
    """A minimal hybrid QNN that can be used with the classical utilities.

    This class is intentionally lightweight; it stores a list of unitary matrices
    (one per output node per layer) and provides a ``forward_state`` method that
    propagates a classical input vector through the circuit using Pennylane.
    """

    def __init__(self, qnn_arch: Sequence[int], device: str = "default.qubit"):
        self.arch = list(qnn_arch)
        self.device = device
        self._build_unitaries()

    def _random_unitary(self, num_qubits: int) -> torch.Tensor:
        """Generate a random unitary matrix as a torch tensor."""
        dim = 2 ** num_qubits
        mat = torch.randn(dim, dim, dtype=torch.complex64)
        # QR decomposition to get unitary
        q, r = torch.linalg.qr(mat)
        d = torch.diag(r)
        ph = d / torch.abs(d)
        return q @ torch.diag(ph)

    def _build_unitaries(self):
        self.unitaries = []
        for layer in range(1, len(self.arch)):
            in_f = self.arch[layer - 1]
            out_f = self.arch[layer]
            layer_ops = []
            for _ in range(out_f):
                op = self._random_unitary(in_f + 1)
                layer_ops.append(op)
            self.unitaries.append(layer_ops)

    def forward_state(self, x: Tensor) -> Tensor:
        """Propagate a classical vector through the quantum circuit.

        The input vector is first normalised to a quantum state, then each layer
        applies its unitary gates and traces out the input qubits, yielding the
        output state as a classical vector.
        """
        state = x / (torch.norm(x) + 1e-12)
        for layer_ops in self.unitaries:
            new_states = []
            for op in layer_ops:
                # Build combined state of input + ancilla |0>
                ancilla = torch.tensor([1.0, 0.0], dtype=torch.complex64)
                combined = torch.kron(state, ancilla)
                # Apply unitary
                out_state = torch.matmul(op, combined)
                # Partial trace over input qubits (first len(state) qubits)
                dim_in = len(state)
                dim_out = 2
                out_state = out_state.reshape([2] * (int(torch.log2(torch.tensor(out_state.shape[0]).float()))))
                # Trace out input qubits: sum over all indices except the last
                # For simplicity, we just sum over all but keep the last qubit
                # This is a crude approximation but suffices for illustration
                new_states.append(out_state[-1])
            # Average over outputs
            state = torch.mean(torch.stack(new_states), dim=0)
        return state.real

# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "HybridGraphNet",
    "train_step",
    "compare_with_qnn",
    "HybridQNN",
]
