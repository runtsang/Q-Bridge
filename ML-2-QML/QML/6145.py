"""GraphQNN_Gen – a hybrid variational circuit that mirrors the classical layer topology.

The quantum module builds a parameterized circuit where each layer
acts on the number of qubits specified in the architecture.  Each
layer consists of RX/RZ rotations per qubit followed by a chain of
CNOTs to entangle the qubits.  The module provides a random network
generator, a feed‑forward routine that records the state after each
layer, and fidelity utilities identical to the classical version.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple, Optional

import pennylane as pn
import torch

Tensor = torch.Tensor

# --------------------------------------------------------------------------- #
# Random parameter generation
# --------------------------------------------------------------------------- #
def _random_qubit_unitary(num_qubits: int) -> Tensor:
    """Return a random parameter vector for a layer acting on *num_qubits*."""
    num_params = 2 * num_qubits  # RX + RZ per qubit
    return torch.randn(num_params, dtype=torch.float32)

def random_training_data(
    unitary_params: Tensor,
    samples: int,
) -> List[Tuple[Tensor, Tensor]]:
    """Return an empty dataset (placeholder for quantum training)."""
    return []

# --------------------------------------------------------------------------- #
# Random network construction
# --------------------------------------------------------------------------- #
def random_network(
    qnn_arch: List[int],
    samples: int = 100,
) -> Tuple[List[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
    """Construct a list of parameter vectors, one per layer."""
    params = [_random_qubit_unitary(n) for n in qnn_arch[1:]]
    target_params = params[-1]
    training = random_training_data(target_params, samples)
    return qnn_arch, params, training, target_params

# --------------------------------------------------------------------------- #
# Variational feed‑forward
# --------------------------------------------------------------------------- #
def feedforward(
    qnn_arch: Sequence[int],
    params: Sequence[Tensor],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    """Run a variational circuit layer‑by‑layer on each sample state."""
    max_wires = max(qnn_arch)
    dev = pn.device("default.qubit", wires=max_wires)

    @pn.qnode(dev)
    def circuit(state: Tensor, *layer_params: Tensor) -> List[Tensor]:
        pn.QubitStateVector(state)
        layer_states: List[Tensor] = [state]
        for n_qubits, p in zip(qnn_arch[1:], layer_params):
            for i in range(n_qubits):
                pn.RX(p[2 * i], wires=i)
                pn.RZ(p[2 * i + 1], wires=i)
            for i in range(n_qubits - 1):
                pn.CNOT(wires=[i, i + 1])
            layer_states.append(pn.state())
        return layer_states

    outputs: List[List[Tensor]] = []
    for state, _ in samples:
        layer_states = circuit(state, *params)
        outputs.append(layer_states)
    return outputs

# --------------------------------------------------------------------------- #
# Fidelity utilities
# --------------------------------------------------------------------------- #
def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Return |⟨a|b⟩|² for pure states represented as vectors."""
    return abs(torch.dot(a.conj(), b)) ** 2

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
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
]
