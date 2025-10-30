"""GraphQNN__gen207: Quantum graph‑neural‑network utilities.

This module implements the quantum counterpart of the classical
GraphQNN.  It provides a variational circuit that mirrors the
classical feed‑forward, a dataset generator that samples random
states, and a benchmark harness that compares the learned unitary
against the ground truth via state fidelity.
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import numpy as np
import qutip as qt

Tensor = qt.Qobj


def _tensored_identity(num_qubits: int) -> qt.Qobj:
    I = qt.qeye(2 ** num_qubits)
    dims = [2] * num_qubits
    I.dims = [dims.copy(), dims.copy()]
    return I


def _tensored_zero(num_qubits: int) -> qt.Qobj:
    zero = qt.fock(2 ** num_qubits).proj()
    dims = [2] * num_qubits
    zero.dims = [dims.copy(), dims.copy()]
    return zero


def _swap_registers(op: qt.Qobj, src: int, tgt: int) -> qt.Qobj:
    if src == tgt:
        return op
    order = list(range(len(op.dims[0])))
    order[src], order[tgt] = order[tgt], order[src]
    return op.permute(order)


def _random_unitary(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    mat = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    q, _ = np.linalg.qr(mat)
    u = qt.Qobj(q)
    dims = [2] * num_qubits
    u.dims = [dims.copy(), dims.copy()]
    return u


def _random_state(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    vec = np.random.randn(dim) + 1j * np.random.randn(dim)
    vec /= np.linalg.norm(vec)
    state = qt.Qobj(vec)
    dims = [2] * num_qubits
    state.dims = [[2] * num_qubits, [1] * num_qubits]
    return state


class GraphQNN:
    """Quantum graph‑neural‑network that learns a target unitary.

    Parameters
    ----------
    architecture : Sequence[int]
        Widths of each layer; the last entry gives the number of qubits
        in the output register.
    """

    def __init__(self, architecture: Sequence[int]) -> None:
        self.arch = list(architecture)

    def _layer_channel(
        self,
        layer: int,
        unitaries: Sequence[Sequence[qt.Qobj]],
        input_state: qt.Qobj,
    ) -> qt.Qobj:
        """Apply a layer of gates to ``input_state`` and trace out the
        input register."""
        num_inputs = self.arch[layer - 1]
        num_outputs = self.arch[layer]
        state = qt.tensor(input_state, _tensored_zero(num_outputs))
        layer_unitary = unitaries[layer][0]
        for gate in unitaries[layer][1:]:
            layer_unitary = gate * layer_unitary
        out_state = layer_unitary * state * layer_unitary.dag()
        if num_inputs!= 0:
            keep = list(range(num_inputs, num_inputs + num_outputs))
            out_state = out_state.ptrace(keep)
        return out_state

    def feedforward(
        self,
        unitaries: Sequence[Sequence[qt.Qobj]],
        samples: Iterable[Tuple[qt.Qobj, qt.Qobj]],
    ) -> List[List[qt.Qobj]]:
        """Return a list of state trajectories for all samples."""
        results: List[List[qt.Qobj]] = []
        for inp, _ in samples:
            traj = [inp]
            current = inp
            for layer in range(1, len(self.arch)):
                current = self._layer_channel(layer, unitaries, current)
                traj.append(current)
            results.append(traj)
        return results

    @staticmethod
    def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
        """Squared overlap between two pure states."""
        return abs((a.dag() * b)[0, 0]) ** 2

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[qt.Qobj],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Build a weighted graph from state fidelities."""
        G = nx.Graph()
        G.add_nodes_from(range(len(states)))
        for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNN.state_fidelity(a, b)
            if fid >= threshold:
                G.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                G.add_edge(i, j, weight=secondary_weight)
        return G

    @classmethod
    def random_network(
        cls,
        architecture: Sequence[int],
        samples: int,
    ) -> Tuple["GraphQNN", List[Tuple[qt.Qobj, qt.Qobj]], qt.Qobj]:
        """Generate a random target unitary and a dataset that maps
        input states to the unitary‑transformed states."""
        target = _random_unitary(architecture[-1])
        dataset = [( _random_state(architecture[0]), target * _random_state(architecture[0]) ) for _ in range(samples)]
        return cls(architecture), dataset, target

    @staticmethod
    def benchmark(
        arch: Sequence[int],
        unitaries: Sequence[Sequence[qt.Qobj]],
        samples: Iterable[Tuple[qt.Qobj, qt.Qobj]],
        target: qt.Qobj,
    ) -> dict:
        """Run a forward pass and report fidelity, average loss, and time."""
        start = time.time()
        fidelities = []
        losses = []
        qnn = GraphQNN(arch)
        for inp, tgt in samples:
            traj = qnn.feedforward(unitaries, [(inp, tgt)])
            out = traj[0][-1]
            fid = GraphQNN.state_fidelity(out, tgt)
            fidelities.append(fid)
            losses.append(1 - fid)
        elapsed = time.time() - start
        return {
            "avg_fidelity": sum(fidelities) / len(fidelities),
            "avg_loss": sum(losses) / len(losses),
            "time_sec": elapsed,
        }
