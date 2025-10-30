"""Hybrid quantum module providing a variational classifier, sampler, and graph utilities.

The quantum functions mirror the classical counterparts: ``build_classifier_circuit``
produces a Qiskit circuit with a data‑encoding layer, a configurable depth of
variational layers, and a measurement head.  ``SamplerQNN`` builds a small
parameterised circuit that can be used with Qiskit Machine Learning’s
``SamplerQNN`` wrapper.  Graph utilities operate on qutip Qobj states and
generate fidelity‑based adjacency graphs.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List, Sequence

import networkx as nx
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
import qutip as qt
import scipy as sc
import itertools
from typing import Sequence

# --------------------------------------------------------------------------- #
# 1. Variational classifier circuit
# --------------------------------------------------------------------------- #

def build_classifier_circuit(
    num_qubits: int,
    depth: int,
    *,
    entanglement: str = "linear",
) -> Tuple[QuantumCircuit, Iterable[ParameterVector], Iterable[ParameterVector], List[SparsePauliOp]]:
    """
    Construct a parameterised quantum circuit that mirrors the classical
    depth‑wise feed‑forward network.

    Parameters
    ----------
    num_qubits:
        Number of qubits / input features.
    depth:
        Number of variational layers.
    entanglement:
        Pattern of two‑qubit entangling gates: ``"linear"`` (CZ between neighbours)
        or ``"full"`` (CZ between every pair).

    Returns
    -------
    circuit:
        Qiskit QuantumCircuit instance.
    encoding:
        ParameterVector for the data encoding.
    weights:
        ParameterVector for the variational parameters.
    observables:
        List of PauliZ operators used to read out the classification logits.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)

    # Data encoding
    for i, param in enumerate(encoding):
        circuit.rx(param, i)

    # Variational layers
    idx = 0
    for _ in range(depth):
        # One‑qubit rotations
        for q in range(num_qubits):
            circuit.ry(weights[idx], q)
            idx += 1
        # Entangling gates
        if entanglement == "full":
            for q1 in range(num_qubits):
                for q2 in range(q1 + 1, num_qubits):
                    circuit.cz(q1, q2)
        else:  # linear
            for q in range(num_qubits - 1):
                circuit.cz(q, q + 1)

    # Measurement observables
    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]

    return circuit, list(encoding), list(weights), observables


# --------------------------------------------------------------------------- #
# 2. Quantum sampler circuit
# --------------------------------------------------------------------------- #

def SamplerQNN() -> qiskit_machine_learning.neural_networks.SamplerQNN:
    """
    Build a small parameterised quantum circuit that can be wrapped
    by Qiskit Machine Learning's ``SamplerQNN`` class.

    Returns
    -------
    qiskit_machine_learning.neural_networks.SamplerQNN
        The sampler ready for integration with the Qiskit ML API.
    """
    from qiskit_machine_learning.neural_networks import SamplerQNN
    from qiskit.primitives import StatevectorSampler as Sampler

    inputs = ParameterVector("input", 2)
    weights = ParameterVector("weight", 4)

    qc = QuantumCircuit(2)
    qc.ry(inputs[0], 0)
    qc.ry(inputs[1], 1)
    qc.cx(0, 1)
    qc.ry(weights[0], 0)
    qc.ry(weights[1], 1)
    qc.cx(0, 1)
    qc.ry(weights[2], 0)
    qc.ry(weights[3], 1)

    sampler = Sampler()
    return SamplerQNN(circuit=qc, input_params=inputs, weight_params=weights, sampler=sampler)


# --------------------------------------------------------------------------- #
# 3. Graph utilities (qutip based)
# --------------------------------------------------------------------------- #

def _tensored_id(num_qubits: int) -> qt.Qobj:
    """Return an identity operator on ``num_qubits`` qubits."""
    identity = qt.qeye(2 ** num_qubits)
    dims = [2] * num_qubits
    identity.dims = [dims.copy(), dims.copy()]
    return identity


def _tensored_zero(num_qubits: int) -> qt.Qobj:
    """Return the projector onto the all‑zero computational basis state."""
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


def random_training_data(unitary: qt.Qobj, samples: int) -> List[Tuple[qt.Qobj, qt.Qobj]]:
    dataset: List[Tuple[qt.Qobj, qt.Qobj]] = []
    num_qubits = len(unitary.dims[0])
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        dataset.append((state, unitary * state))
    return dataset


def random_network(qnn_arch: List[int], samples: int):
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    unitaries: List[List[qt.Qobj]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        layer_ops: List[qt.Qobj] = []
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


def _layer_channel(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[qt.Qobj]],
    layer: int,
    input_state: qt.Qobj,
) -> qt.Qobj:
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    state = qt.tensor(input_state, _tensored_zero(num_outputs))

    layer_unitary = unitaries[layer][0].copy()
    for gate in unitaries[layer][1:]:
        layer_unitary = gate * layer_unitary

    return _partial_trace_remove(layer_unitary * state * layer_unitary.dag(), range(num_inputs))


def feedforward(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[qt.Qobj]],
    samples: Iterable[Tuple[qt.Qobj, qt.Qobj]],
) -> List[List[qt.Qobj]]:
    stored_states: List[List[qt.Qobj]] = []
    for sample, _ in samples:
        layerwise = [sample]
        current_state = sample
        for layer in range(1, len(qnn_arch)):
            current_state = _layer_channel(qnn_arch, unitaries, layer, current_state)
            layerwise.append(current_state)
        stored_states.append(layerwise)
    return stored_states


def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
    return abs((a.dag() * b)[0, 0]) ** 2


def fidelity_adjacency(
    states: Sequence[qt.Qobj],
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
    "build_classifier_circuit",
    "SamplerQNN",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
]
