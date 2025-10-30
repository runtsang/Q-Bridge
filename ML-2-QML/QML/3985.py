"""Graph‑based hybrid neural network – Quantum (QuTiP + Qiskit) implementation.

This module matches the public interface of the classical version,
but propagates quantum states through parameterised unitaries.
A Qiskit SamplerQNN is provided for measurement‑based sampling."""
from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import qutip as qt
import scipy as sc
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN as QiskitSamplerQNN
from qiskit.primitives import StatevectorSampler

Tensor = qt.Qobj


def _tensored_identity(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    iden = qt.qeye(dim)
    iden.dims = [[2] * num_qubits, [2] * num_qubits]
    return iden


def _tensored_zero(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    zero = qt.fock(dim).proj()
    zero.dims = [[2] * num_qubits, [2] * num_qubits]
    return zero


def _swap_registers(op: qt.Qobj, src: int, tgt: int) -> qt.Qobj:
    if src == tgt:
        return op
    order = list(range(len(op.dims[0])))
    order[src], order[tgt] = order[tgt], order[src]
    return op.permute(order)


def _random_qubit_unitary(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    mat = sc.random.normal(size=(dim, dim)) + 1j * sc.random.normal(size=(dim, dim))
    unitary = sc.linalg.orth(mat)
    qobj = qt.Qobj(unitary)
    qobj.dims = [[2] * num_qubits, [2] * num_qubits]
    return qobj


def _random_qubit_state(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    vec = sc.random.normal(size=(dim, 1)) + 1j * sc.random.normal(size=(dim, 1))
    vec /= sc.linalg.norm(vec)
    qobj = qt.Qobj(vec)
    qobj.dims = [[2] * num_qubits, [1] * num_qubits]
    return qobj


def _random_training_data(target: qt.Qobj, samples: int) -> List[Tuple[qt.Qobj, qt.Qobj]]:
    """Generate (state, target_state) training pairs."""
    data: List[Tuple[qt.Qobj, qt.Qobj]] = []
    n_qubits = len(target.dims[0])
    for _ in range(samples):
        state = _random_qubit_state(n_qubits)
        data.append((state, target * state))
    return data


class GraphQuantumNeuralNetworkQutip:
    """Quantum graph neural network with fidelity‑based graph utilities."""

    def __init__(self, arch: Sequence[int], seed: int | None = None) -> None:
        self.arch = tuple(arch)
        self.seed = seed
        if seed is not None:
            sc.random.seed(seed)
        # Layerwise unitaries: each output qubit gets its own unitary.
        self.unitaries: List[List[qt.Qobj]] = [[]]
        for layer in range(1, len(arch)):
            in_q = arch[layer - 1]
            out_q = arch[layer]
            ops: List[qt.Qobj] = []
            for out_idx in range(out_q):
                op = _random_qubit_unitary(in_q + 1)
                if out_q > 1:
                    op = qt.tensor(_random_qubit_unitary(in_q + 1), _tensored_identity(out_q - 1))
                    op = _swap_registers(op, in_q, in_q + out_idx)
                ops.append(op)
            self.unitaries.append(ops)
        self.target_unitary = self.unitaries[-1][0]  # last layer first gate

    def random_training_data(self, samples: int) -> List[Tuple[qt.Qobj, qt.Qobj]]:
        return _random_training_data(self.target_unitary, samples)

    def feedforward(
        self, samples: Iterable[Tuple[qt.Qobj, qt.Qobj]]
    ) -> List[List[qt.Qobj]]:
        """Return state after each layer for every sample."""
        states: List[List[qt.Qobj]] = []
        for inp, _ in samples:
            layer_states = [inp]
            current = inp
            for layer in range(1, len(self.arch)):
                # prepend ancilla zeros for the new outputs
                state = qt.tensor(current, _tensored_zero(self.arch[layer] - self.arch[layer - 1]))
                unitary = self.unitaries[layer][0].copy()
                for gate in self.unitaries[layer][1:]:
                    unitary = gate * unitary
                current = _partial_trace(state, unitary, self.arch[layer - 1])
                layer_states.append(current)
            states.append(layer_states)
        return states

    @staticmethod
    def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
        """Squared overlap of pure states."""
        return abs((a.dag() * b)[0, 0]) ** 2

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[qt.Qobj],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        G = nx.Graph()
        G.add_nodes_from(range(len(states)))
        for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
            fid = GraphQuantumNeuralNetworkQutip.state_fidelity(a, b)
            if fid >= threshold:
                G.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                G.add_edge(i, j, weight=secondary_weight)
        return G

    # ------------------------------------------------------------------
    # Quantum sampler via Qiskit – analogous to classical SamplerQNN
    # ------------------------------------------------------------------
    def build_sampler(self, shots: int = 1024) -> QiskitSamplerQNN:
        """Return a Qiskit SamplerQNN that samples from the last layer."""
        # Build a parameterised circuit from the unitaries
        num_qubits = self.arch[-1]
        inputs = ParameterVector("input", num_qubits)
        weights = ParameterVector("weight", sum(len(l) for l in self.unitaries))
        qc = QuantumCircuit(num_qubits)
        # Map the custom unitaries to parameterised gates (simplified)
        for layer, ops in enumerate(self.unitaries[1:], start=1):
            for gate in ops:
                # Use a generic Ry rotation for demonstration
                qc.ry(weights.pop(0), 0)
        # Instantiate the sampler
        sampler = StatevectorSampler()
        return QiskitSamplerQNN(circuit=qc, input_params=inputs, weight_params=weights, sampler=sampler)

__all__ = [
    "GraphQuantumNeuralNetworkQutip",
    "build_sampler",
]
