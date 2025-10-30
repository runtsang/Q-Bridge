"""HybridEstimatorQNN – quantum‑centric implementation using Qiskit.

The class re‑implements the same public API as the classical version but
computes the kernel value by encoding classical data into a quantum
state with Ry gates and evaluating overlap via Qiskit’s statevector
simulator.  The rest of the pipeline (classical NN and fidelity graph)
is identical to the classical variant, enabling a direct comparison.

"""

from __future__ import annotations

import itertools
from typing import Sequence

import networkx as nx
import numpy as np
import torch
from torch import nn
from qiskit import Aer, execute
from qiskit.circuit import QuantumCircuit

# --------------------------------------------------------------------------- #
# 1) Classical neural network – same as in the classical module
# --------------------------------------------------------------------------- #
class _ClassicNN(nn.Module):
    """Compact fully‑connected regressor."""

    def __init__(self, in_features: int, hidden_sizes: Sequence[int], out_features: int = 1) -> None:
        super().__init__()
        layers = []
        last = in_features
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.Tanh())
            last = h
        layers.append(nn.Linear(last, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)

    def get_hidden(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        activations = [inputs]
        x = inputs
        for layer in self.net:
            x = layer(x)
            activations.append(x)
        return activations

# --------------------------------------------------------------------------- #
# 2) Quantum kernel – Qiskit implementation
# --------------------------------------------------------------------------- #
class _QuantumKernelQiskit:
    """Compute a quantum kernel using Qiskit state‑vector simulator."""

    def __init__(self, n_wires: int) -> None:
        self.n_wires = n_wires
        self.backend = Aer.get_backend("statevector_simulator")

    def _encode(self, qc: QuantumCircuit, vector: np.ndarray) -> None:
        for i, val in enumerate(vector):
            qc.ry(val, i)

    def kernel(self, x: np.ndarray, y: np.ndarray) -> float:
        """Return |<x|y>|^2 where |x> and |y> are encoded by Ry gates."""
        qc_x = QuantumCircuit(self.n_wires)
        self._encode(qc_x, x)
        sv_x = execute(qc_x, self.backend).result().get_statevector()
        qc_y = QuantumCircuit(self.n_wires)
        self._encode(qc_y, y)
        sv_y = execute(qc_y, self.backend).result().get_statevector()
        return float(np.abs(np.vdot(sv_x, sv_y)) ** 2)

# --------------------------------------------------------------------------- #
# 3) Fidelity utilities – same as in the classical module
# --------------------------------------------------------------------------- #
def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float((a_norm @ b_norm).item() ** 2)

def fidelity_adjacency(
    states: Sequence[torch.Tensor],
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

# --------------------------------------------------------------------------- #
# 4) Hybrid estimator – classical NN + Qiskit kernel + graph
# --------------------------------------------------------------------------- #
class HybridEstimatorQNN:
    """Hybrid classical‑quantum estimator with Qiskit kernel."""

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: Sequence[int] = (8, 4),
        graph_threshold: float = 0.8,
    ) -> None:
        self.nn = _ClassicNN(
            in_features=input_dim + 1,
            hidden_sizes=hidden_sizes,
            out_features=1,
        )
        self.kernel = _QuantumKernelQiskit(n_wires=input_dim)
        self.ref_vector = np.zeros(input_dim)
        self.graph_threshold = graph_threshold

    def _kernel_values(self, inputs: np.ndarray) -> np.ndarray:
        return np.array([self.kernel.kernel(x, self.ref_vector) for x in inputs])

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        kv = self._kernel_values(inputs).reshape(-1, 1)
        extended = np.concatenate([inputs, kv], axis=1)
        return self.nn(torch.tensor(extended, dtype=torch.float32)).detach().numpy()

    def extract_last_hidden(self, inputs: np.ndarray) -> torch.Tensor:
        kv = self._kernel_values(inputs).reshape(-1, 1)
        extended = np.concatenate([inputs, kv], axis=1)
        activations = self.nn.get_hidden(torch.tensor(extended, dtype=torch.float32))
        return activations[-2]  # shape (batch, features)

    def build_fidelity_graph(self, inputs: np.ndarray) -> nx.Graph:
        last_layer = self.extract_last_hidden(inputs)
        states = [last_layer[i] for i in range(last_layer.shape[0])]
        return fidelity_adjacency(
            states,
            self.graph_threshold,
        )

__all__ = ["HybridEstimatorQNN"]
