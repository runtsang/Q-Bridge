"""Quantum graph neural network for patch‑wise processing.

This module implements a lightweight graph‑based quantum neural net
mirroring the structure of the `GraphQNN.py` seed.  Each node in the
graph represents a 2×2 image patch; the node state is a pure
qubit state derived from the patch values.  Random unitary layers
propagate information across the graph, and fidelity‑based adjacency
matrices are constructed for downstream analysis or hybrid training.
"""

from __future__ import annotations

import itertools
from typing import Iterable, Sequence, Tuple, List

import networkx as nx
import qutip as qt
import scipy as sc

# --------------------------------------------------------------------------- #
#  Core quantum utilities
# --------------------------------------------------------------------------- #

def _tensored_id(num_qubits: int) -> qt.Qobj:
    """Return a tensor‑product identity of `num_qubits` qubits."""
    identity = qt.qeye(2 ** num_qubits)
    dims = [2] * num_qubits
    identity.dims = [dims.copy(), dims.copy()]
    return identity

def _tensored_zero(num_qubits: int) -> qt.Qobj:
    """Return a projector onto |0…0⟩ for `num_qubits` qubits."""
    projector = qt.fock(2 ** num_qubits).proj()
    dims = [2] * num_qubits
    projector.dims = [dims.copy(), dims.copy()]
    return projector

def _swap_registers(op: qt.Qobj, source: int, target: int) -> qt.Qobj:
    """Swap the registers at positions `source` and `target` in `op`."""
    if source == target:
        return op
    order = list(range(len(op.dims[0])))
    order[source], order[target] = order[target], order[source]
    return op.permute(order)

def _random_qubit_unitary(num_qubits: int) -> qt.Qobj:
    """Generate a random Haar‑distributed unitary on `num_qubits`."""
    dim = 2 ** num_qubits
    matrix = sc.random.normal(size=(dim, dim)) + 1j * sc.random.normal(size=(dim, dim))
    unitary = sc.linalg.orth(matrix)
    qobj = qt.Qobj(unitary)
    dims = [2] * num_qubits
    qobj.dims = [dims.copy(), dims.copy()]
    return qobj

def _random_qubit_state(num_qubits: int) -> qt.Qobj:
    """Sample a random pure state on `num_qubits` qubits."""
    dim = 2 ** num_qubits
    amplitudes = sc.random.normal(size=(dim, 1)) + 1j * sc.random.normal(size=(dim, 1))
    amplitudes /= sc.linalg.norm(amplitudes)
    state = qt.Qobj(amplitudes)
    state.dims = [[2] * num_qubits, [1] * num_qubits]
    return state

# --------------------------------------------------------------------------- #
#  Dataset generation
# --------------------------------------------------------------------------- #

def random_training_data(unitary: qt.Qobj, samples: int) -> List[Tuple[qt.Qobj, qt.Qobj]]:
    """
    Generate training pairs (input_state, target_state) where
    target_state = unitary * input_state.
    """
    dataset: List[Tuple[qt.Qobj, qt.Qobj]] = []
    num_qubits = len(unitary.dims[0])
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        dataset.append((state, unitary * state))
    return dataset

def random_network(qnn_arch: List[int], samples: int):
    """
    Build a random quantum graph neural network.

    Parameters
    ----------
    qnn_arch : List[int]
        Layer widths.  The first element is the number of input nodes
        (patches).  Each subsequent element is the number of output
        nodes for that layer.
    samples : int
        Number of training samples to generate.

    Returns
    -------
    Tuple[List[int], List[List[qt.Qobj]], List[Tuple[qt.Qobj, qt.Qobj]], qt.Qobj]
        Architecture, list of per‑layer unitary lists, training data,
        and the target unitary of the final layer.
    """
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    unitaries: List[List[qt.Qobj]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        layer_ops: List[qt.Qobj] = []
        for output in range(num_outputs):
            # Each output node receives a fresh unitary acting on the
            # concatenation of its inputs and the new output qubit.
            op = _random_qubit_unitary(num_inputs + 1)
            if num_outputs > 1:
                op = qt.tensor(_random_qubit_unitary(num_inputs + 1), _tensored_id(num_outputs - 1))
                op = _swap_registers(op, num_inputs, num_inputs + output)
            layer_ops.append(op)
        unitaries.append(layer_ops)

    return qnn_arch, unitaries, training_data, target_unitary

# --------------------------------------------------------------------------- #
#  Forward propagation
# --------------------------------------------------------------------------- #

def _partial_trace_keep(state: qt.Qobj, keep: Sequence[int]) -> qt.Qobj:
    if len(keep)!= len(state.dims[0]):
        return state.ptrace(list(keep))
    return state

def _partial_trace_remove(state: qt.Qobj, remove: Sequence[int]) -> qt.Qobj:
    keep = list(range(len(state.dims[0])))
    for index in sorted(remove, reverse=True):
        keep.pop(index)
    return _partial_trace_keep(state, keep)

def _layer_channel(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qt.Qobj]],
                   layer: int, input_state: qt.Qobj) -> qt.Qobj:
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    # Prepare the joint state: input + fresh |0…0⟩ qubits
    state = qt.tensor(input_state, _tensored_zero(num_outputs))

    # Build the composite unitary for this layer
    layer_unitary = unitaries[layer][0].copy()
    for gate in unitaries[layer][1:]:
        layer_unitary = gate * layer_unitary

    # Apply and trace out the input qubits
    new_state = layer_unitary * state * layer_unitary.dag()
    return _partial_trace_remove(new_state, range(num_inputs))

def feedforward(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qt.Qobj]],
                samples: Iterable[Tuple[qt.Qobj, qt.Qobj]]):
    """
    Propagate each sample through the network and record the state at every layer.

    Returns
    -------
    List[List[qt.Qobj]]
        Outer list over samples, inner list over layers.
    """
    stored_states: List[List[qt.Qobj]] = []
    for sample, _ in samples:
        layerwise = [sample]
        current_state = sample
        for layer in range(1, len(qnn_arch)):
            current_state = _layer_channel(qnn_arch, unitaries, layer, current_state)
            layerwise.append(current_state)
        stored_states.append(layerwise)
    return stored_states

# --------------------------------------------------------------------------- #
#  Graph utilities
# --------------------------------------------------------------------------- #

def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
    """Absolute squared overlap between two pure states."""
    return abs((a.dag() * b)[0, 0]) ** 2

def fidelity_adjacency(states: Sequence[qt.Qobj], threshold: float,
                       *, secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
    """
    Build a weighted graph where nodes are states and edges indicate
    high fidelity.

    Parameters
    ----------
    states : Sequence[qt.Qobj]
        List of pure states.
    threshold : float
        Primary fidelity threshold for edge inclusion.
    secondary : float | None
        Secondary threshold for weaker edges.
    secondary_weight : float
        Weight for secondary edges.

    Returns
    -------
    networkx.Graph
        Weighted adjacency graph.
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

__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
]
