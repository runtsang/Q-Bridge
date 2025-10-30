"""GraphQNN__gen110_qml - quantum module with variational circuit and fidelity graph."""

from __future__ import annotations

import itertools
from typing import Iterable, Sequence

import networkx as nx
import qutip as qt
import scipy as sc
import numpy as np
import torch
import pennylane as qml_lib

# --- Original QNN state propagation -------------------------------------------------


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
    dataset = []
    num_qubits = len(unitary.dims[0])
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        dataset.append((state, unitary * state))
    return dataset


def random_network(qnn_arch: list[int], samples: int):
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


def _partial_trace_keep(state: qt.Qobj, keep: Sequence[int]) -> qt.Qobj:
    if len(keep)!= len(state.dims[0]):
        return state.ptrace(list(keep))
    return state


def _partial_trace_remove(state: qt.Qobj, remove: Sequence[int]) -> qt.Qobj:
    keep = list(range(len(state.dims[0])))
    for index in sorted(remove, reverse=True):
        keep.pop(index)
    return _partial_trace_keep(state, keep)


def _layer_channel(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qt.Qobj]], layer: int, input_state: qt.Qobj) -> qt.Qobj:
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    state = qt.tensor(input_state, _tensored_zero(num_outputs))

    layer_unitary = unitaries[layer][0].copy()
    for gate in unitaries[layer][1:]:
        layer_unitary = gate * layer_unitary

    return _partial_trace_remove(layer_unitary * state * layer_unitary.dag(), range(num_inputs))


def feedforward(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qt.Qobj]], samples: Iterable[tuple[qt.Qobj, qt.Qobj]]):
    stored_states = []
    for sample, _ in samples:
        layerwise = [sample]
        current_state = sample
        for layer in range(1, len(qnn_arch)):
            current_state = _layer_channel(qnn_arch, unitaries, layer, current_state)
            layerwise.append(current_state)
        stored_states.append(layerwise)
    return stored_states


# --- Fidelity utilities ------------------------------------------------------------


def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
    """Return the absolute squared overlap between pure states ``a`` and ``b``."""
    return abs((a.dag() * b)[0, 0]) ** 2


def fidelity_adjacency(states: Sequence[qt.Qobj], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
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


# --- Variational circuit infrastructure --------------------------------------------

# Number of qubits for the variational circuit
NUM_QUBITS: int = 2

# PennyLane device
dev = qml_lib.device("default.qubit", wires=NUM_QUBITS)


@qml_lib.qnode(dev, interface="torch")
def var_circuit_torch(params: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """
    Variational circuit with a single RY rotation per qubit followed by a CNOT
    between qubits 0 and 1 (if NUM_QUBITS > 1).
    """
    # Prepare the input state
    qml_lib.StatePrep(inputs, wires=range(NUM_QUBITS))
    # Apply parameterized rotations
    for i in range(NUM_QUBITS):
        qml_lib.RY(params[i], wires=i)
    # Entangling layer
    if NUM_QUBITS > 1:
        qml_lib.CNOT(wires=[0, 1])
    # Return the state vector
    return qml_lib.state()


def apply_variational_circuit(params: torch.Tensor, input_state: qt.Qobj) -> qt.Qobj:
    """
    Run the variational circuit on a qutip state and return the resulting
    qutip Qobj.  The function is differentiable with respect to ``params``.
    """
    # Convert qutip state to torch vector
    input_vec = torch.tensor(input_state.full().flatten(), dtype=torch.complex64)
    # Execute the circuit
    output_vec = var_circuit_torch(params, input_vec)
    # Convert back to qutip
    output_state = qt.Qobj(output_vec.detach().numpy(), dims=[[2] * NUM_QUBITS, [1] * NUM_QUBITS])
    return output_state


def compute_fidelity(output_state: torch.Tensor, target_vec: torch.Tensor) -> torch.Tensor:
    """
    Compute the fidelity between two state vectors in torch.
    """
    return torch.abs(torch.dot(output_state, target_vec.conj())) ** 2


__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "_tensored_id",
    "_tensored_zero",
    "_swap_registers",
    "_random_qubit_unitary",
    "_random_qubit_state",
    "apply_variational_circuit",
    "compute_fidelity",
    "NUM_QUBITS",
]
