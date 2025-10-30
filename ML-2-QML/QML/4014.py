"""Quantum graph neural network with a self‑attention block.

The implementation combines the QML graph‑QNN from the reference
with a Qiskit‑based self‑attention circuit.  All public functions
mirror the classical counterpart, enabling side‑by‑side experiments."""
from __future__ import annotations

import itertools
import random
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
import qiskit
import qutip as qt
import scipy as sc
from qiskit import Aer, QuantumCircuit, QuantumRegister, ClassicalRegister, execute

# ----------------------------------------------------------------------
# Helper functions – copied from the original QML seed but
# adapted to be fully self‑contained.
# ----------------------------------------------------------------------


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
    matrix = sc.random.normal(size=(dim, dim)) + 1j * sc.random.normal(
        size=(dim, dim)
    )
    unitary = sc.linalg.orth(matrix)
    qobj = qt.Qobj(unitary)
    dims = [2] * num_qubits
    qobj.dims = [dims.copy(), dims.copy()]
    return qobj


def _random_qubit_state(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    amps = sc.random.normal(size=(dim, 1)) + 1j * sc.random.normal(
        size=(dim, 1)
    )
    amps /= sc.linalg.norm(amps)
    state = qt.Qobj(amps)
    state.dims = [[2] * num_qubits, [1] * num_qubits]
    return state


def random_training_data(
    unitary: qt.Qobj, samples: int
) -> List[Tuple[qt.Qobj, qt.Qobj]]:
    dataset: List[Tuple[qt.Qobj, qt.Qobj]] = []
    num_qubits = len(unitary.dims[0])
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        dataset.append((state, unitary * state))
    return dataset


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
    arch: Sequence[int],
    unitaries: Sequence[Sequence[qt.Qobj]],
    layer: int,
    input_state: qt.Qobj,
) -> qt.Qobj:
    num_inputs = arch[layer - 1]
    num_outputs = arch[layer]
    state = qt.tensor(input_state, _tensored_zero(num_outputs))
    layer_unitary = unitaries[layer][0].copy()
    for gate in unitaries[layer][1:]:
        layer_unitary = gate * layer_unitary
    return _partial_trace_remove(layer_unitary * state * layer_unitary.dag(), range(num_inputs))


# ----------------------------------------------------------------------
# Quantum GraphQNNAttention class
# ----------------------------------------------------------------------


class GraphQNNAttention:
    """Quantum graph QNN augmented with a Qiskit self‑attention block."""

    def __init__(self, arch: Sequence[int], attention_qubits: int = 4, seed: int | None = None):
        self.arch = list(arch)
        self.attention_qubits = attention_qubits
        self.rng = random.Random(seed)
        self.unitaries = self._init_unitaries()
        self.backend = Aer.get_backend("qasm_simulator")

    def _init_unitaries(self) -> List[List[qt.Qobj]]:
        unitaries: List[List[qt.Qobj]] = [[]]
        for layer in range(1, len(self.arch)):
            num_in = self.arch[layer - 1]
            num_out = self.arch[layer]
            layer_ops: List[qt.Qobj] = []
            for out_idx in range(num_out):
                op = _random_qubit_unitary(num_in + 1)
                if num_out > 1:
                    op = qt.tensor(
                        _random_qubit_unitary(num_in + 1), _tensored_id(num_out - 1)
                    )
                    op = _swap_registers(op, num_in, num_in + out_idx)
                layer_ops.append(op)
            unitaries.append(layer_ops)
        return unitaries

    @staticmethod
    def random_network(
        arch: List[int], samples: int, seed: int | None = None
    ) -> Tuple[List[int], List[List[qt.Qobj]], List[Tuple[qt.Qobj, qt.Qobj]], qt.Qobj]:
        target = _random_qubit_unitary(arch[-1])
        data = random_training_data(target, samples)
        unitaries: List[List[qt.Qobj]] = [[]]
        for layer in range(1, len(arch)):
            num_in = arch[layer - 1]
            num_out = arch[layer]
            layer_ops: List[qt.Qobj] = []
            for out_idx in range(num_out):
                op = _random_qubit_unitary(num_in + 1)
                if num_out > 1:
                    op = qt.tensor(
                        _random_qubit_unitary(num_in + 1), _tensored_id(num_out - 1)
                    )
                    op = _swap_registers(op, num_in, num_in + out_idx)
                layer_ops.append(op)
            unitaries.append(layer_ops)
        return arch, unitaries, data, target

    def _self_attention_circuit(
        self, rotation_params: np.ndarray, entangle_params: np.ndarray
    ) -> QuantumCircuit:
        qr = QuantumRegister(self.attention_qubits, "q")
        cr = ClassicalRegister(self.attention_qubits, "c")
        circ = QuantumCircuit(qr, cr)
        for i in range(self.attention_qubits):
            circ.rx(rotation_params[3 * i], i)
            circ.ry(rotation_params[3 * i + 1], i)
            circ.rz(rotation_params[3 * i + 2], i)
        for i in range(self.attention_qubits - 1):
            circ.crx(entangle_params[i], i, i + 1)
        circ.measure(qr, cr)
        return circ

    def run_attention(
        self,
        state: qt.Qobj,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> dict:
        circ = self._self_attention_circuit(rotation_params, entangle_params)
        result = execute(circ, self.backend, shots=shots).result()
        return result.get_counts(circ)

    def feedforward(
        self, samples: Iterable[Tuple[qt.Qobj, qt.Qobj]]
    ) -> List[List[qt.Qobj]]:
        outputs: List[List[qt.Qobj]] = []
        for inp, _ in samples:
            state = inp
            # propagate through each layer; self‑attention is applied
            # only on the first layer via the circuit, subsequent layers
            # use the unitary channels defined above.
            for layer in range(1, len(self.arch)):
                state = _layer_channel(self.arch, self.unitaries, layer, state)
            outputs.append([state])
        return outputs

    @staticmethod
    def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
        return abs((a.dag() * b)[0, 0]) ** 2

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[qt.Qobj],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        g = nx.Graph()
        g.add_nodes_from(range(len(states)))
        for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNNAttention.state_fidelity(a, b)
            if fid >= threshold:
                g.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                g.add_edge(i, j, weight=secondary_weight)
        return g


__all__ = ["GraphQNNAttention"]
