"""Quantum graph‑convolutional network inspired by QCNN, using graph edges to guide entanglement."""

from __future__ import annotations

import itertools
from typing import Iterable, Sequence, Tuple, List

import networkx as nx
import qiskit as qk
import qiskit.quantum_info as qi
import qiskit.circuit as qc
import qiskit.circuit.library as qcl
import numpy as np
import scipy as sc
import qutip as qt

# --------------------------------------------------------------------------- #
#  Core QCNN primitives (adapted from the original QCNN seed)
# --------------------------------------------------------------------------- #

def _conv_circuit(params: qk.circuit.ParameterVector) -> qc.QuantumCircuit:
    """Two‑qubit convolution block used in QCNN layers."""
    circ = qc.QuantumCircuit(2)
    circ.rz(-np.pi / 2, 1)
    circ.cx(1, 0)
    circ.rz(params[0], 0)
    circ.ry(params[1], 1)
    circ.cx(0, 1)
    circ.ry(params[2], 1)
    circ.cx(1, 0)
    circ.rz(np.pi / 2, 0)
    return circ


def _pool_circuit(params: qk.circuit.ParameterVector) -> qc.QuantumCircuit:
    """Two‑qubit pooling block used in QCNN layers."""
    circ = qc.QuantumCircuit(2)
    circ.rz(-np.pi / 2, 1)
    circ.cx(1, 0)
    circ.rz(params[0], 0)
    circ.ry(params[1], 1)
    circ.cx(0, 1)
    circ.ry(params[2], 1)
    return circ


def conv_layer(num_qubits: int, param_prefix: str) -> qc.QuantumCircuit:
    """Convolutional layer that applies `_conv_circuit` to adjacent qubit pairs."""
    circ = qc.QuantumCircuit(num_qubits)
    params = qc.circuit.ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
        circ.append(_conv_circuit(params[q1 * 3 : q1 * 3 + 3]), [q1, q2])
    return circ


def pool_layer(sources: List[int], sinks: List[int], param_prefix: str) -> qc.QuantumCircuit:
    """Pooling layer that reduces the qubit count from `sources` to `sinks`."""
    num_qubits = len(sources) + len(sinks)
    circ = qc.QuantumCircuit(num_qubits)
    params = qc.circuit.ParameterVector(param_prefix, length=(num_qubits // 2) * 3)
    for src, snk in zip(sources, sinks):
        circ.append(_pool_circuit(params[(src // 2) * 3 : (src // 2) * 3 + 3]), [src, snk])
    return circ


# --------------------------------------------------------------------------- #
#  Graph‑aware random network generator
# --------------------------------------------------------------------------- #

def _random_qubit_unitary(num_qubits: int) -> qt.Qobj:
    """Random unitary on `num_qubits` qubits."""
    dim = 2 ** num_qubits
    mat = sc.random.normal(size=(dim, dim)) + 1j * sc.random.normal(size=(dim, dim))
    unitary = sc.linalg.orth(mat)
    return qt.Qobj(unitary, dims=[[2] * num_qubits, [2] * num_qubits])


def _random_qubit_state(num_qubits: int) -> qt.Qobj:
    """Random pure state on `num_qubits` qubits."""
    dim = 2 ** num_qubits
    vec = sc.random.normal(size=(dim, 1)) + 1j * sc.random.normal(size=(dim, 1))
    vec /= sc.linalg.norm(vec)
    return qt.Qobj(vec, dims=[[2] * num_qubits, [1] * num_qubits])


def random_training_data(unitary: qt.Qobj, samples: int) -> List[Tuple[qt.Qobj, qt.Qobj]]:
    """Generate input–target pairs for training the quantum network."""
    data: List[Tuple[qt.Qobj, qt.Qobj]] = []
    n = len(unitary.dims[0])
    for _ in range(samples):
        state = _random_qubit_state(n)
        data.append((state, unitary * state))
    return data


def random_network(
    qnn_arch: List[int], samples: int, adjacency: nx.Graph
) -> Tuple[List[int], List[List[qt.Qobj]], List[Tuple[qt.Qobj, qt.Qobj]], qt.Qobj]:
    """
    Create a QCNN‑style network where entanglement patterns follow the graph edges.

    Parameters
    ----------
    qnn_arch
        List of qubit counts per layer (must match adjacency size).
    samples
        Number of training samples.
    adjacency
        Graph that determines which qubit pairs interact in convolution blocks.
    """
    # Target unitary is a random unitary on the final layer’s qubits
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    # Build layer‑wise unitaries; each layer is a list of gates acting on qubit pairs
    layer_ops: List[List[qt.Qobj]] = [[]]
    for layer_idx in range(1, len(qnn_arch)):
        n_inputs = qnn_arch[layer_idx - 1]
        n_outputs = qnn_arch[layer_idx]
        ops_this_layer: List[qt.Qobj] = []

        # Iterate over edges in the subgraph induced by the current layer
        subgraph = adjacency.subgraph(range(n_inputs))
        for u, v in subgraph.edges():
            # Each edge receives its own random 2‑qubit unitary
            op = _random_qubit_unitary(2)
            # Embed into the full space: tensor with identity on other qubits
            op_full = qt.tensor(op, _tensored_id(n_outputs - 2))
            # Move the two qubits to the correct positions
            op_full = _swap_registers(op_full, 0, u)
            op_full = _swap_registers(op_full, 1, v)
            ops_this_layer.append(op_full)

        # If the layer is not fully connected, pad with identities
        if len(ops_this_layer) < n_inputs:
            ops_this_layer.extend([_tensored_id(n_outputs)] * (n_inputs - len(ops_this_layer)))

        layer_ops.append(ops_this_layer)

    return qnn_arch, layer_ops, training_data, target_unitary


# --------------------------------------------------------------------------- #
#  Partial‑trace helpers for state propagation
# --------------------------------------------------------------------------- #

def _tensored_id(num_qubits: int) -> qt.Qobj:
    """Identity on `num_qubits` qubits with correct dims."""
    I = qt.qeye(2 ** num_qubits)
    I.dims = [[2] * num_qubits, [2] * num_qubits]
    return I


def _tensored_zero(num_qubits: int) -> qt.Qobj:
    """Zero projector on `num_qubits` qubits."""
    zero = qt.fock(2 ** num_qubits).proj()
    zero.dims = [[2] * num_qubits, [2] * num_qubits]
    return zero


def _swap_registers(op: qt.Qobj, source: int, target: int) -> qt.Qobj:
    """Swap two qubits in `op`’s ordering."""
    if source == target:
        return op
    order = list(range(len(op.dims[0])))
    order[source], order[target] = order[target], order[source]
    return op.permute(order)


def _partial_trace_remove(state: qt.Qobj, remove: Sequence[int]) -> qt.Qobj:
    """Trace out qubits specified by `remove`."""
    keep = list(range(len(state.dims[0])))
    for idx in sorted(remove, reverse=True):
        keep.pop(idx)
    return state.ptrace(keep)


def _layer_channel(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[qt.Qobj]],
    layer: int,
    input_state: qt.Qobj,
) -> qt.Qobj:
    """Apply one convolution/pooling layer and trace out the discarded qubits."""
    n_inputs = qnn_arch[layer - 1]
    n_outputs = qnn_arch[layer]
    # Pad the input with zeros for the new qubits introduced in this layer
    state = qt.tensor(input_state, _tensored_zero(n_outputs))

    # Compose all gates belonging to this layer
    layer_unitary = _tensored_id(n_outputs)
    for gate in unitaries[layer]:
        layer_unitary = gate * layer_unitary

    # Apply the unitary and trace out the input qubits
    return _partial_trace_remove(layer_unitary * state * layer_unitary.dag(), range(n_inputs))


def feedforward(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[qt.Qobj]],
    samples: Iterable[Tuple[qt.Qobj, qt.Qobj]],
) -> List[List[qt.Qobj]]:
    """Propagate a batch of states through the QCNN‑style network."""
    all_states: List[List[qt.Qobj]] = []
    for sample, _ in samples:
        states = [sample]
        current = sample
        for layer in range(1, len(qnn_arch)):
            current = _layer_channel(qnn_arch, unitaries, layer, current)
            states.append(current)
        all_states.append(states)
    return all_states


# --------------------------------------------------------------------------- #
#  Fidelity helpers (identical to the classical version)
# --------------------------------------------------------------------------- #

def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
    """Return the absolute squared overlap between pure states `a` and `b`."""
    return abs((a.dag() * b)[0, 0]) ** 2


def fidelity_adjacency(
    states: Sequence[qt.Qobj],
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


__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
]
