"""GraphQNNGen: classical graph‑neural‑network framework.

The module exposes a single class `GraphQNN` that can run in
either classical or quantum mode, while keeping the same public
API – `random_network`, `feedforward`, `fidelity_adjacency` and
`estimate_from_adjacency`.  The classical implementation relies on
PyTorch; the quantum implementation is delegated to Qiskit and
Qiskit‑Machine‑Learning.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor

# ---------- Classical utilities --------------------------------------------

def _random_linear(in_features: int, out_features: int) -> Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)

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

# ---------- Estimator helper -----------------------------------------------

def EstimatorQNN() -> nn.Module:
    """Return a tiny fully‑connected regression network."""
    class Estimator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 8),
                nn.Tanh(),
                nn.Linear(8, 4),
                nn.Tanh(),
                nn.Linear(4, 1),
            )
        def forward(self, inputs: Tensor) -> Tensor:  # type: ignore[override]
            return self.net(inputs)
    return Estimator()

# ---------- Hybrid class -----------------------------------------------------

class GraphQNN:
    """Unified classical/quantum graph‑neural‑network interface.

    Parameters
    ----------
    arch : Sequence[int]
        Layer sizes, e.g. ``[2, 8, 4, 1]``.
    mode : str, optional
        ``'classical'`` (default) or ``'quantum'``.  The quantum mode
        lazily imports Qiskit when first used.
    """
    def __init__(self, arch: Sequence[int], mode: str = "classical") -> None:
        self.arch = list(arch)
        self.mode = mode
        if mode == "classical":
            self.weights = [_random_linear(in_f, out_f)
                            for in_f, out_f in zip(self.arch[:-1], self.arch[1:])]
        else:
            # quantum mode uses Qiskit; weights are not classical tensors
            self._init_quantum()

    # ------------------------------------------------------------------ quantum

    def _init_quantum(self) -> None:
        import qiskit
        from qiskit.circuit import Parameter
        from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
        from qiskit.primitives import Estimator as QiskitEstimator
        from qiskit.quantum_info import SparsePauliOp

        # Build a simple variational circuit that mirrors the classical
        # architecture.  Each layer is a single qubit with Ry/ Rz rotations
        # parametrised by the layer size.
        total_qubits = sum(self.arch)
        total_params = 2 * total_qubits
        params = [Parameter(f"p{i}") for i in range(total_params)]

        qc = qiskit.QuantumCircuit(total_qubits)
        for q in range(total_qubits):
            qc.ry(params[q], q)
            qc.rz(params[total_qubits + q], q)

        observable = SparsePauliOp.from_list([("Z" * qc.num_qubits, 1)])
        estimator = QiskitEstimator()
        self.estimator_qnn = QiskitEstimatorQNN(
            circuit=qc,
            observables=observable,
            input_params=params[:total_qubits],
            weight_params=params[total_qubits:],
            estimator=estimator,
        )

    # ------------------------------------------------------------------ public API

    def random_network(self, samples: int):
        if self.mode == "classical":
            return random_network(self.arch, samples)
        else:
            # For quantum mode, reuse the seed code logic
            from qiskit import QuantumCircuit
            import qutip as qt
            import scipy as sc
            # Simplified: generate a random unitary for the whole network
            dim = 2 ** sum(self.arch)
            matrix = sc.random.normal(size=(dim, dim)) + 1j * sc.random.normal(size=(dim, dim))
            unitary = sc.linalg.orth(matrix)
            target = qt.Qobj(unitary)
            return self.arch, target, [], target

    def feedforward(self, samples: Iterable[Tuple[Tensor, Tensor]] | List[tuple[qt.Qobj, qt.Qobj]]):
        if self.mode == "classical":
            return feedforward(self.arch, self.weights, samples)
        else:
            # Evaluate the quantum EstimatorQNN on each input state
            outputs = []
            for state, _ in samples:
                # Convert qutip state to vector for Qiskit
                vec = np.array(state.data).flatten()
                out = self.estimator_qnn.predict(vec)
                outputs.append([state] + out)
            return outputs

    def fidelity_adjacency(
        self,
        states: Sequence[Tensor] | Sequence[qt.Qobj],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        if self.mode == "classical":
            return fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)
        else:
            # quantum fidelity
            def q_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
                return abs((a.dag() * b)[0, 0]) ** 2
            graph = nx.Graph()
            graph.add_nodes_from(range(len(states)))
            for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
                fid = q_fidelity(a, b)
                if fid >= threshold:
                    graph.add_edge(i, j, weight=1.0)
                elif secondary is not None and fid >= secondary:
                    graph.add_edge(i, j, weight=secondary_weight)
            return graph

    def estimate_from_adjacency(self, graph: nx.Graph, device: str | None = None) -> nn.Module:
        """Return a tiny estimator trained on node‑pair features."""
        # Convert graph edges into a feature matrix
        edges = list(graph.edges(data=True))
        if not edges:
            raise ValueError("Graph has no edges to learn from.")
        X = torch.tensor([[w["weight"], i, j] for i, j, w in edges], dtype=torch.float32)
        y = torch.tensor([w["weight"] for _, _, w in edges], dtype=torch.float32).unsqueeze(-1)

        estimator = EstimatorQNN()
        optimizer = torch.optim.Adam(estimator.parameters(), lr=0.01)
        for _ in range(200):
            optimizer.zero_grad()
            loss = F.mse_loss(estimator(X), y)
            loss.backward()
            optimizer.step()
        return estimator

__all__ = [
    "GraphQNN",
    "random_network",
    "feedforward",
    "fidelity_adjacency",
    "EstimatorQNN",
]
