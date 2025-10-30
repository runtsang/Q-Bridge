"""
GraphQNN: Quantum implementation using Qiskit.

Features
--------
* Random network generation with parameterised unitaries.
* Feed‑forward propagation of quantum states through each layer.
* State‑vector fidelity and weighted graph construction.
* Quantum kernel based on state‑vector fidelity.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import qiskit
from qiskit.quantum_info import Operator, Statevector, random_unitary, state_fidelity

# --------------------------------------------------------------------------- #
# Core utilities
# --------------------------------------------------------------------------- #
def _tensored_id(num_qubits: int) -> Operator:
    """Return the identity operator on ``num_qubits`` qubits."""
    return Operator(qiskit.quantum_info.Operator.from_label("I" * num_qubits))


def _tensored_zero(num_qubits: int) -> Operator:
    """Return the |0⟩⟨0| projector on ``num_qubits`` qubits."""
    zero = Statevector.from_label("0" * num_qubits)
    return Operator(zero.projection())


def _random_qubit_unitary(num_qubits: int) -> Operator:
    """Generate a random unitary on ``num_qubits`` qubits."""
    return Operator(random_unitary(2 ** num_qubits))


def _random_qubit_state(num_qubits: int) -> Statevector:
    """Generate a random pure state on ``num_qubits`` qubits."""
    return Statevector.random(num_qubits)


def random_training_data(
    target_unitary: Operator,
    samples: int,
) -> List[Tuple[Statevector, Statevector]]:
    """Generate (input, target) state pairs for the target unitary."""
    data: List[Tuple[Statevector, Statevector]] = []
    for _ in range(samples):
        inp = _random_qubit_state(target_unitary.dims[0][0])
        out = target_unitary @ inp
        data.append((inp, out))
    return data


def random_network(
    qnn_arch: List[int],
    samples: int,
) -> Tuple[List[int], List[List[Operator]], List[Tuple[Statevector, Statevector]], Operator]:
    """
    Return (architecture, list of per‑layer unitaries, training data, target unitary).
    Each layer contains a list of unitaries that act on its input qubits plus one ancilla.
    """
    num_qubits = qnn_arch[0]
    unitaries: List[List[Operator]] = [[]]
    for layer in range(1, len(qnn_arch)):
        prev = qnn_arch[layer - 1]
        out = qnn_arch[layer]
        layer_ops: List[Operator] = []

        for _ in range(out):
            # Each output qubit gets a random unitary on (prev + 1) qubits
            layer_ops.append(_random_qubit_unitary(prev + 1))

        unitaries.append(layer_ops)

    # Target unitary is just a random unitary on the final number of qubits
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)
    return qnn_arch, unitaries, training_data, target_unitary


def _layer_channel(
    arch: Sequence[int],
    unitaries: Sequence[Sequence[Operator]],
    layer: int,
    input_state: Statevector,
) -> Statevector:
    """Apply a single layer of unitaries to the input state."""
    prev = arch[layer - 1]
    out = arch[layer]
    # Pad input with |0⟩ ancilla qubits for each output
    ancilla = _tensored_zero(out)
    state = input_state.tensor(ancilla)

    # Compose all unitaries in the layer
    layer_unitary = unitaries[layer][0]
    for gate in unitaries[layer][1:]:
        layer_unitary = layer_unitary @ gate

    # Apply the unitary and trace out ancilla
    new_state = layer_unitary @ state
    # Trace out ancilla qubits
    keep = list(range(prev))
    return new_state.truncate(keep)


def feedforward(
    arch: Sequence[int],
    unitaries: Sequence[Sequence[Operator]],
    samples: Iterable[Tuple[Statevector, Statevector]],
) -> List[List[Statevector]]:
    """Propagate each sample through all layers and record the intermediate states."""
    stored: List[List[Statevector]] = []
    for inp, _ in samples:
        layerwise = [inp]
        current = inp
        for layer in range(1, len(arch)):
            current = _layer_channel(arch, unitaries, layer, current)
            layerwise.append(current)
        stored.append(layerwise)
    return stored


def state_fidelity(a: Statevector, b: Statevector) -> float:
    """Return the squared modulus of the inner product."""
    return abs((a @ b.conj()).data[0]) ** 2


def fidelity_adjacency(
    states: Sequence[Statevector],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Create a weighted graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


def quantum_kernel(
    state_a: Statevector,
    state_b: Statevector,
    *,
    eps: float = 1e-12,
) -> float:
    """Return the fidelity kernel between two states."""
    return state_fidelity(state_a, state_b)


__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "quantum_kernel",
]
