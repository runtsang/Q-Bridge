"""GraphQNN: Quantum implementation with extended capabilities.

Features:
* Gate sharing across layers.
* Batch‑wise feed‑forward.
* Optional noise in synthetic data.
* Unified GraphQNN class that mirrors the classical API.
"""

import itertools
from typing import Iterable, List, Sequence, Tuple, Dict, Optional

import networkx as nx
import qutip as qt
import scipy as sc

__all__ = ["GraphQNN"]


class GraphQNN:
    """Quantum graph neural network with extended utilities."""

    def __init__(self, arch: Sequence[int], unitaries: Optional[List[List[qt.Qobj]]] = None):
        """
        Parameters
        ----------
        arch : Sequence[int]
            Architecture: number of qubits per layer.
        unitaries : Optional[List[List[qt.Qobj]]], default=None
            Nested list of unitaries per layer. If ``None`` random unitaries are created.
        """
        self.arch = list(arch)
        if unitaries is None:
            self.unitaries = self._random_network_units()
        else:
            self.unitaries = unitaries

    @staticmethod
    def _tensored_id(num_qubits: int) -> qt.Qobj:
        """Identity operator on ``num_qubits`` qubits."""
        dim = 2 ** num_qubits
        I = qt.qeye(dim)
        I.dims = [[2] * num_qubits, [2] * num_qubits]
        return I

    @staticmethod
    def _tensored_zero(num_qubits: int) -> qt.Qobj:
        """Zero projector on ``num_qubits`` qubits."""
        dim = 2 ** num_qubits
        zero = qt.fock(dim, 0).proj()
        zero.dims = [[2] * num_qubits, [2] * num_qubits]
        return zero

    @staticmethod
    def _swap_registers(op: qt.Qobj, source: int, target: int) -> qt.Qobj:
        """Swap qubits ``source`` and ``target`` within operator ``op``."""
        if source == target:
            return op
        order = list(range(len(op.dims[0])))
        order[source], order[target] = order[target], order[source]
        return op.permute(order)

    @staticmethod
    def _random_qubit_unitary(num_qubits: int) -> qt.Qobj:
        """Generate a random unitary on ``num_qubits`` qubits."""
        dim = 2 ** num_qubits
        mat = sc.random.normal(size=(dim, dim)) + 1j * sc.random.normal(size=(dim, dim))
        qobj = qt.Qobj(sc.linalg.orth(mat))
        qobj.dims = [[2] * num_qubits, [2] * num_qubits]
        return qobj

    @staticmethod
    def _random_qubit_state(num_qubits: int) -> qt.Qobj:
        """Generate a random pure state on ``num_qubits`` qubits."""
        dim = 2 ** num_qubits
        vec = sc.random.normal(size=(dim, 1)) + 1j * sc.random.normal(size=(dim, 1))
        vec /= sc.linalg.norm(vec)
        qobj = qt.Qobj(vec)
        qobj.dims = [[2] * num_qubits, [1] * num_qubits]
        return qobj

    @staticmethod
    def random_training_data(
        unitary: qt.Qobj,
        samples: int,
        *,
        noise: float = 0.0,
    ) -> List[Tuple[qt.Qobj, qt.Qobj]]:
        """
        Generate synthetic training pairs ``(psi, U psi)``.
        Optional additive noise can be applied to the output state.
        """
        dataset: List[Tuple[qt.Qobj, qt.Qobj]] = []
        num_qubits = len(unitary.dims[0])
        for _ in range(samples):
            psi = GraphQNN._random_qubit_state(num_qubits)
            out = unitary * psi
            if noise > 0.0:
                out = out + noise * qt.rand_unitary(num_qubits) * psi
            dataset.append((psi, out))
        return dataset

    @staticmethod
    def random_network(
        arch: Sequence[int], samples: int
    ) -> Tuple[List[int], List[List[qt.Qobj]], List[Tuple[qt.Qobj, qt.Qobj]], qt.Qobj]:
        """
        Create a random network of unitary layers and synthetic training data
        for the final layer.
        """
        target_unitary = GraphQNN._random_qubit_unitary(arch[-1])
        training_data = GraphQNN.random_training_data(target_unitary, samples)
        unitaries: List[List[qt.Qobj]] = [[]]
        for layer in range(1, len(arch)):
            num_in = arch[layer - 1]
            num_out = arch[layer]
            layer_ops: List[qt.Qobj] = []
            for _ in range(num_out):
                op = GraphQNN._random_qubit_unitary(num_in + 1)
                if num_out > 1:
                    op = qt.tensor(op, GraphQNN._tensored_id(num_out - 1))
                    op = GraphQNN._swap_registers(op, num_in, num_in + _)
                layer_ops.append(op)
            unitaries.append(layer_ops)
        return list(arch), unitaries, training_data, target_unitary

    def _random_network_units(self) -> List[List[qt.Qobj]]:
        """Generate a random network of unitaries for each layer."""
        units: List[List[qt.Qobj]] = [[]]
        for layer in range(1, len(self.arch)):
            num_in = self.arch[layer - 1]
            num_out = self.arch[layer]
            layer_ops: List[qt.Qobj] = []
            for _ in range(num_out):
                op = GraphQNN._random_qubit_unitary(num_in + 1)
                if num_out > 1:
                    op = qt.tensor(op, GraphQNN._tensored_id(num_out - 1))
                    op = GraphQNN._swap_registers(op, num_in, num_in + _)
                layer_ops.append(op)
            units.append(layer_ops)
        return units

    def _partial_trace_keep(self, state: qt.Qobj, keep: Sequence[int]) -> qt.Qobj:
        """Partial trace over all qubits except those in ``keep``."""
        if len(keep) == len(state.dims[0]):
            return state
        return state.ptrace(list(keep))

    def _partial_trace_remove(self, state: qt.Qobj, remove: Sequence[int]) -> qt.Qobj:
        """Partial trace over qubits specified in ``remove``."""
        keep = list(range(len(state.dims[0])))
        for idx in sorted(remove, reverse=True):
            keep.pop(idx)
        return self._partial_trace_keep(state, keep)

    def _layer_channel(
        self,
        layer: int,
        input_state: qt.Qobj,
    ) -> qt.Qobj:
        """Apply the unitary channel of ``layer`` to ``input_state``."""
        num_in = self.arch[layer - 1]
        num_out = self.arch[layer]
        state = qt.tensor(input_state, GraphQNN._tensored_zero(num_out))
        layer_unitary = self.unitaries[layer][0]
        for gate in self.unitaries[layer][1:]:
            layer_unitary = gate * layer_unitary
        return self._partial_trace_remove(layer_unitary * state * layer_unitary.dag(), range(num_in))

    def feedforward(
        self, samples: Iterable[Tuple[qt.Qobj, qt.Qobj]]
    ) -> List[List[qt.Qobj]]:
        """Forward pass for each sample in ``samples``."""
        activations: List[List[qt.Qobj]] = []
        for psi, _ in samples:
            states = [psi]
            current = psi
            for layer in range(1, len(self.arch)):
                current = self._layer_channel(layer, current)
                states.append(current)
            activations.append(states)
        return activations

    def batch_feedforward(self, batch: List[qt.Qobj]) -> List[qt.Qobj]:
        """Forward pass for a batch of input states."""
        outputs: List[qt.Qobj] = []
        for psi in batch:
            out = psi
            for layer in range(1, len(self.arch)):
                out = self._layer_channel(layer, out)
            outputs.append(out)
        return outputs

    @staticmethod
    def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
        """Squared overlap of two pure states."""
        return abs((a.dag() * b)[0, 0]) ** 2

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[qt.Qobj],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Build a graph from state fidelities with weighted edges."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNN.state_fidelity(a, b)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    def share_unitaries(self, layer_indices: List[int]) -> None:
        """Share the first unitary across the specified layers."""
        if not layer_indices:
            return
        base_unitary = self.unitaries[0][0]
        for idx in layer_indices:
            if 0 <= idx < len(self.unitaries):
                self.unitaries[idx] = [base_unitary] + self.unitaries[idx][1:]
