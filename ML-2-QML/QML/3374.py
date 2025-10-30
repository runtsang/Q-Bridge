"""Quantum graph neural network module with hybrid self‑attention.

The quantum implementation mirrors the classical interface but uses
Qiskit and QuTiP to build and propagate unitary layers.  A quantum
self‑attention circuit is also provided; it can be run on a simulator
or a real device.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer
import qutip as qt
import scipy as sc

Tensor = qt.Qobj


def _tensored_id(num_qubits: int) -> qt.Qobj:
    """Identity operator with explicit qubit dimensions."""
    id_ = qt.qeye(2 ** num_qubits)
    dims = [2] * num_qubits
    id_.dims = [dims.copy(), dims.copy()]
    return id_


def _tensored_zero(num_qubits: int) -> qt.Qobj:
    """Projector onto the all‑zero computational basis state."""
    zero = qt.fock(2 ** num_qubits).proj()
    dims = [2] * num_qubits
    zero.dims = [dims.copy(), dims.copy()]
    return zero


def _swap_registers(op: qt.Qobj, src: int, tgt: int) -> qt.Qobj:
    """Permute qubit registers inside a tensor product."""
    if src == tgt:
        return op
    order = list(range(len(op.dims[0])))
    order[src], order[tgt] = order[tgt], order[src]
    return op.permute(order)


def _random_qubit_unitary(num_qubits: int) -> qt.Qobj:
    """Generate a Haar‑random unitary on `num_qubits` qubits."""
    dim = 2 ** num_qubits
    mat = sc.random.normal(size=(dim, dim)) + 1j * sc.random.normal(size=(dim, dim))
    u = sc.linalg.orth(mat)
    qobj = qt.Qobj(u)
    dims = [2] * num_qubits
    qobj.dims = [dims.copy(), dims.copy()]
    return qobj


def _random_qubit_state(num_qubits: int) -> qt.Qobj:
    """Sample a normalized pure state on `num_qubits` qubits."""
    dim = 2 ** num_qubits
    amp = sc.random.normal(size=(dim, 1)) + 1j * sc.random.normal(size=(dim, 1))
    amp /= sc.linalg.norm(amp)
    state = qt.Qobj(amp)
    state.dims = [[2] * num_qubits, [1] * num_qubits]
    return state


def random_training_data(unitary: qt.Qobj, samples: int) -> List[Tuple[qt.Qobj, qt.Qobj]]:
    """Generate training pairs (|ψ⟩, U|ψ⟩)."""
    data: List[Tuple[qt.Qobj, qt.Qobj]] = []
    n = len(unitary.dims[0])
    for _ in range(samples):
        s = _random_qubit_state(n)
        data.append((s, unitary * s))
    return data


def random_network(qnn_arch: List[int], samples: int):
    """Instantiate a random quantum network and its training set."""
    target = _random_qubit_unitary(qnn_arch[-1])
    training = random_training_data(target, samples)

    unitaries: List[List[qt.Qobj]] = [[]]
    for layer in range(1, len(qnn_arch)):
        in_ = qnn_arch[layer - 1]
        out_ = qnn_arch[layer]
        layer_ops: List[qt.Qobj] = []
        for out_i in range(out_):
            op = _random_qubit_unitary(in_ + 1)
            if out_ > 1:
                op = qt.tensor(_random_qubit_unitary(in_ + 1), _tensored_id(out_ - 1))
                op = _swap_registers(op, in_, in_ + out_i)
            layer_ops.append(op)
        unitaries.append(layer_ops)

    return qnn_arch, unitaries, training, target


def _partial_trace_keep(state: qt.Qobj, keep: Sequence[int]) -> qt.Qobj:
    """Trace out all qubits except those in `keep`."""
    if len(keep)!= len(state.dims[0]):
        return state.ptrace(list(keep))
    return state


def _partial_trace_remove(state: qt.Qobj, remove: Sequence[int]) -> qt.Qobj:
    """Trace out qubits listed in `remove`."""
    keep = list(range(len(state.dims[0])))
    for idx in sorted(remove, reverse=True):
        keep.pop(idx)
    return _partial_trace_keep(state, keep)


def _layer_channel(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[qt.Qobj]],
    layer: int,
    input_state: qt.Qobj,
) -> qt.Qobj:
    """Propagate a state through a single layer of the quantum network."""
    in_ = qnn_arch[layer - 1]
    out_ = qnn_arch[layer]
    state = qt.tensor(input_state, _tensored_zero(out_))

    unitary = unitaries[layer][0].copy()
    for gate in unitaries[layer][1:]:
        unitary = gate * unitary

    return _partial_trace_remove(unitary * state * unitary.dag(), range(in_))


def feedforward(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[qt.Qobj]],
    samples: Iterable[Tuple[qt.Qobj, qt.Qobj]],
) -> List[List[qt.Qobj]]:
    """Compute layer‑wise quantum states for a batch of samples."""
    states: List[List[qt.Qobj]] = []
    for s, _ in samples:
        layer_states = [s]
        cur = s
        for layer in range(1, len(qnn_arch)):
            cur = _layer_channel(qnn_arch, unitaries, layer, cur)
            layer_states.append(cur)
        states.append(layer_states)
    return states


def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
    """Squared absolute overlap between two pure states."""
    return abs((a.dag() * b)[0, 0]) ** 2


def fidelity_adjacency(
    states: Sequence[qt.Qobj],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Build a weighted graph from state fidelities."""
    g = nx.Graph()
    g.add_nodes_from(range(len(states)))
    for (i, ai), (j, aj) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(ai, aj)
        if fid >= threshold:
            g.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            g.add_edge(i, j, weight=secondary_weight)
    return g


class QuantumSelfAttention:
    """Quantum circuit implementing a self‑attention style block."""

    def __init__(self, n_qubits: int = 4):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        qc = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            qc.rx(rotation_params[3 * i], i)
            qc.ry(rotation_params[3 * i + 1], i)
            qc.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            qc.crx(entangle_params[i], i, i + 1)
        qc.measure(self.qr, self.cr)
        return qc

    def run(
        self,
        backend,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ):
        qc = self._build(rotation_params, entangle_params)
        job = execute(qc, backend, shots=shots)
        return job.result().get_counts(qc)


class GraphQNNHybrid:
    """Quantum counterpart of GraphQNNHybrid with identical public API."""

    def __init__(self, qnn_arch: List[int]):
        self.arch = list(qnn_arch)
        self.arch, self.unitaries, self.training, self.target = random_network(
            qnn_arch, 10
        )
        self.attention = QuantumSelfAttention(n_qubits=self.arch[-1])

    def feedforward(
        self, samples: Iterable[Tuple[qt.Qobj, qt.Qobj]]
    ) -> List[List[qt.Qobj]]:
        return feedforward(self.arch, self.unitaries, samples)

    def fidelity_graph(
        self,
        states: Sequence[qt.Qobj],
        *args,
        **kwargs,
    ) -> nx.Graph:
        return fidelity_adjacency(states, *args, **kwargs)

    def run_attention(
        self,
        backend,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ):
        return self.attention.run(backend, rotation_params, entangle_params, shots)


__all__ = [
    "GraphQNNHybrid",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
]
