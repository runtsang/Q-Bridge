"""
GraphQNNGen – Quantum hybrid utilities.

This module implements:
* Random quantum graph neural networks and state‑based training data.
* Quantum feed‑forward propagation using tensor‑network techniques.
* Fidelity‑based graph construction with Qobj states.
* Quantum QCNN circuits, Quantum‑NAT inspired fully‑connected module,
  and quantum auto‑encoder using Qiskit simulators.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple, Optional

import networkx as nx
import numpy as np
import qiskit as qk
import qutip as qt
import scipy as sc
import torch
import torch.nn as nn

QObj = qt.Qobj
Tensor = torch.Tensor


# ---------------------------------------------------------------------------
# Random quantum network generation
# ---------------------------------------------------------------------------

def _tensored_id(num_qubits: int) -> QObj:
    """Identity operator on `num_qubits` qubits."""
    identity = qt.qeye(2 ** num_qubits)
    dims = [2] * num_qubits
    identity.dims = [dims.copy(), dims.copy()]
    return identity


def _tensored_zero(num_qubits: int) -> QObj:
    """Zero projector on `num_qubits` qubits."""
    projector = qt.fock(2 ** num_qubits).proj()
    dims = [2] * num_qubits
    projector.dims = [dims.copy(), dims.copy()]
    return projector


def _swap_registers(op: QObj, source: int, target: int) -> QObj:
    if source == target:
        return op
    order = list(range(len(op.dims[0])))
    order[source], order[target] = order[target], order[source]
    return op.permute(order)


def _random_qubit_unitary(num_qubits: int) -> QObj:
    dim = 2 ** num_qubits
    matrix = sc.random.normal(size=(dim, dim)) + 1j * sc.random.normal(size=(dim, dim))
    unitary = sc.linalg.orth(matrix)
    qobj = qt.Qobj(unitary)
    dims = [2] * num_qubits
    qobj.dims = [dims.copy(), dims.copy()]
    return qobj


def _random_qubit_state(num_qubits: int) -> QObj:
    dim = 2 ** num_qubits
    amplitudes = sc.random.normal(size=(dim, 1)) + 1j * sc.random.normal(size=(dim, 1))
    amplitudes /= sc.linalg.norm(amplitudes)
    state = qt.Qobj(amplitudes)
    state.dims = [[2] * num_qubits, [1] * num_qubits]
    return state


def random_training_data(unitary: QObj, samples: int) -> list[tuple[QObj, QObj]]:
    """Generate (|ψ⟩, U|ψ⟩) pairs for a given unitary."""
    dataset = []
    num_qubits = len(unitary.dims[0])
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        dataset.append((state, unitary * state))
    return dataset


def random_network_quantum(qnn_arch: list[int], samples: int) -> tuple[list[int], list[list[QObj]], list[tuple[QObj, QObj]], QObj]:
    """Build a layered quantum network with random unitaries."""
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    unitaries: list[list[QObj]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        layer_ops: list[QObj] = []
        for output in range(num_outputs):
            op = _random_qubit_unitary(num_inputs + 1)
            if num_outputs > 1:
                op = qt.tensor(_random_qubit_unitary(num_inputs + 1), _tensored_id(num_outputs - 1))
                op = _swap_registers(op, num_inputs, num_inputs + output)
            layer_ops.append(op)
        unitaries.append(layer_ops)

    return qnn_arch, unitaries, training_data, target_unitary


# ---------------------------------------------------------------------------
# Quantum feed‑forward propagation
# ---------------------------------------------------------------------------

def _partial_trace_keep(state: QObj, keep: Sequence[int]) -> QObj:
    if len(keep)!= len(state.dims[0]):
        return state.ptrace(list(keep))
    return state


def _partial_trace_remove(state: QObj, remove: Sequence[int]) -> QObj:
    keep = list(range(len(state.dims[0])))
    for index in sorted(remove, reverse=True):
        keep.pop(index)
    return _partial_trace_keep(state, keep)


def _layer_channel(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[QObj]], layer: int, input_state: QObj) -> QObj:
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    state = qt.tensor(input_state, _tensored_zero(num_outputs))

    layer_unitary = unitaries[layer][0].copy()
    for gate in unitaries[layer][1:]:
        layer_unitary = gate * layer_unitary

    return _partial_trace_remove(layer_unitary * state * layer_unitary.dag(), range(num_inputs))


def feedforward_quantum(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[QObj]], samples: Iterable[tuple[QObj, QObj]]) -> List[List[QObj]]:
    """Return the state at every layer for each sample."""
    stored_states: List[List[QObj]] = []
    for sample, _ in samples:
        layerwise = [sample]
        current_state = sample
        for layer in range(1, len(qnn_arch)):
            current_state = _layer_channel(qnn_arch, unitaries, layer, current_state)
            layerwise.append(current_state)
        stored_states.append(layerwise)
    return stored_states


# ---------------------------------------------------------------------------
# Fidelity utilities
# ---------------------------------------------------------------------------

def state_fidelity_q(a: QObj, b: QObj) -> float:
    """Squared overlap of two pure Qobj states."""
    return abs((a.dag() * b)[0, 0]) ** 2


def fidelity_adjacency_q(states: Sequence[QObj], threshold: float, *, secondary: Optional[float] = None, secondary_weight: float = 0.5) -> nx.Graph:
    """Create a weighted graph from state fidelities."""
    G = nx.Graph()
    G.add_nodes_from(range(len(states)))
    for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity_q(s_i, s_j)
        if fid >= threshold:
            G.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            G.add_edge(i, j, weight=secondary_weight)
    return G


# ---------------------------------------------------------------------------
# Quantum QCNN construction (Qiskit)
# ---------------------------------------------------------------------------

def _conv_circuit(params: qk.circuit.ParameterVector) -> qk.QuantumCircuit:
    """Two‑qubit convolution block used in the QCNN."""
    qc = qk.QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    qc.cx(1, 0)
    qc.rz(np.pi / 2, 0)
    return qc


def conv_layer(num_qubits: int, param_prefix: str) -> qk.QuantumCircuit:
    """Convolutional layer composed of two‑qubit blocks."""
    qc = qk.QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    params = qk.circuit.ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc = qc.compose(_conv_circuit(params[param_index : param_index + 3]), [q1, q2])
        qc.barrier()
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        qc = qc.compose(_conv_circuit(params[param_index : param_index + 3]), [q1, q2])
        qc.barrier()
        param_index += 3
    qc_inst = qc.to_instruction()
    qc = qk.QuantumCircuit(num_qubits)
    qc.append(qc_inst, qubits)
    return qc


def _pool_circuit(params: qk.circuit.ParameterVector) -> qk.QuantumCircuit:
    """Two‑qubit pooling block."""
    qc = qk.QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc


def pool_layer(sources: Sequence[int], sinks: Sequence[int], param_prefix: str) -> qk.QuantumCircuit:
    """Pooling layer that maps `sources` to `sinks`."""
    num_qubits = len(sources) + len(sinks)
    qc = qk.QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    params = qk.circuit.ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for source, sink in zip(sources, sinks):
        qc = qc.compose(_pool_circuit(params[param_index : param_index + 3]), [source, sink])
        qc.barrier()
        param_index += 3
    qc_inst = qc.to_instruction()
    qc = qk.QuantumCircuit(num_qubits)
    qc.append(qc_inst, range(num_qubits))
    return qc


def QCNN_quantum() -> qk.QuantumCircuit:
    """Return a full QCNN circuit with feature‑map and ansatz."""
    feature_map = qk.circuit.library.ZFeatureMap(8)
    ansatz = qk.QuantumCircuit(8, name="Ansatz")

    # First layer
    ansatz.compose(conv_layer(8, "c1"), inplace=True)

    # First pooling
    ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), inplace=True)

    # Second layer
    ansatz.compose(conv_layer(4, "c2"), inplace=True)

    # Second pooling
    ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), inplace=True)

    # Third layer
    ansatz.compose(conv_layer(2, "c3"), inplace=True)

    # Third pooling
    ansatz.compose(pool_layer([0], [1], "p3"), inplace=True)

    # Combine feature map and ansatz
    circuit = qk.QuantumCircuit(8)
    circuit.compose(feature_map, range(8), inplace=True)
    circuit.compose(ansatz, range(8), inplace=True)

    return circuit


# ---------------------------------------------------------------------------
# Quantum auto‑encoder (Qiskit SamplerQNN)
# ---------------------------------------------------------------------------

def Autoencoder_quantum(num_latent: int = 3, num_trash: int = 2) -> qk.QuantumCircuit:
    """Build a quantum auto‑encoder circuit with a swap‑test."""
    qr = qk.QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = qk.ClassicalRegister(1, "c")
    qc = qk.QuantumCircuit(qr, cr)

    # Ansatz
    qc.compose(qk.circuit.library.RealAmplitudes(num_latent + num_trash, reps=5), range(0, num_latent + num_trash), inplace=True)
    qc.barrier()

    # Swap‑test
    aux = num_latent + 2 * num_trash
    qc.h(aux)
    for i in range(num_trash):
        qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
    qc.h(aux)
    qc.measure(aux, cr[0])
    return qc


# ---------------------------------------------------------------------------
# Quantum‑NAT inspired fully‑connected module (Qiskit)
# ---------------------------------------------------------------------------

def QFCModel_quantum() -> qk.QuantumCircuit:
    """Create a simple Qiskit circuit that imitates the Quantum‑NAT fully‑connected block."""
    qc = qk.QuantumCircuit(4)
    qc.compose(qk.circuit.library.RealAmplitudes(4, reps=2), inplace=True)
    qc.measure_all()
    return qc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "feedforward_quantum",
    "fidelity_adjacency_q",
    "random_network_quantum",
    "random_training_data",
    "state_fidelity_q",
    "QCNN_quantum",
    "Autoencoder_quantum",
    "QFCModel_quantum",
]
