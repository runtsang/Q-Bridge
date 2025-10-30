import numpy as np
import networkx as nx
import itertools
import qiskit as qk
import qutip as qt
import scipy as sc
from typing import Iterable, Sequence, Tuple, List

class GraphQNNGenQML:
    """Quantum counterpart of :class:`GraphQNNGenML`.

    Generates random unitary layers, propagates quantum states,
    builds fidelity‑based adjacency graphs, and provides a
    Qiskit sampler circuit.  The public interface matches the
    classical implementation for easy comparison.
    """

    def __init__(self, arch: Sequence[int]):
        self.arch = list(arch)
        self.unitaries: List[List[qt.Qobj]] = []
        self._build_random_unitaries()

    def _tensored_id(self, num_qubits: int) -> qt.Qobj:
        identity = qt.qeye(2 ** num_qubits)
        dims = [2] * num_qubits
        identity.dims = [dims.copy(), dims.copy()]
        return identity

    def _tensored_zero(self, num_qubits: int) -> qt.Qobj:
        projector = qt.fock(2 ** num_qubits).proj()
        dims = [2] * num_qubits
        projector.dims = [dims.copy(), dims.copy()]
        return projector

    def _swap_registers(self, op: qt.Qobj, source: int, target: int) -> qt.Qobj:
        if source == target:
            return op
        order = list(range(len(op.dims[0])))
        order[source], order[target] = order[target], order[source]
        return op.permute(order)

    def _random_qubit_unitary(self, num_qubits: int) -> qt.Qobj:
        dim = 2 ** num_qubits
        matrix = sc.random.normal(size=(dim, dim)) + 1j * sc.random.normal(size=(dim, dim))
        unitary = sc.linalg.orth(matrix)
        qobj = qt.Qobj(unitary)
        dims = [2] * num_qubits
        qobj.dims = [dims.copy(), dims.copy()]
        return qobj

    def _random_qubit_state(self, num_qubits: int) -> qt.Qobj:
        dim = 2 ** num_qubits
        amplitudes = sc.random.normal(size=(dim, 1)) + 1j * sc.random.normal(size=(dim, 1))
        amplitudes /= sc.linalg.norm(amplitudes)
        state = qt.Qobj(amplitudes)
        state.dims = [[2] * num_qubits, [1] * num_qubits]
        return state

    def _partial_trace_keep(self, state: qt.Qobj, keep: Sequence[int]) -> qt.Qobj:
        if len(keep)!= len(state.dims[0]):
            return state.ptrace(list(keep))
        return state

    def _partial_trace_remove(self, state: qt.Qobj, remove: Sequence[int]) -> qt.Qobj:
        keep = list(range(len(state.dims[0])))
        for idx in sorted(remove, reverse=True):
            keep.pop(idx)
        return self._partial_trace_keep(state, keep)

    def _build_random_unitaries(self) -> None:
        self.unitaries = [[]]
        for layer in range(1, len(self.arch)):
            num_in = self.arch[layer - 1]
            num_out = self.arch[layer]
            ops: List[qt.Qobj] = []
            for out in range(num_out):
                op = self._random_qubit_unitary(num_in + 1)
                if num_out > 1:
                    op = qt.tensor(self._random_qubit_unitary(num_in + 1), self._tensored_id(num_out - 1))
                    op = self._swap_registers(op, num_in, num_in + out)
                ops.append(op)
            self.unitaries.append(ops)

    def random_network(self, samples: int) -> Tuple[List[int], List[List[qt.Qobj]], List[Tuple[qt.Qobj, qt.Qobj]], qt.Qobj]:
        target_unitary = self._random_qubit_unitary(self.arch[-1])
        training_data = self._random_training_data(target_unitary, samples)
        return self.arch, self.unitaries, training_data, target_unitary

    def _random_training_data(self, unitary: qt.Qobj, samples: int) -> List[Tuple[qt.Qobj, qt.Qobj]]:
        dataset: List[Tuple[qt.Qobj, qt.Qobj]] = []
        num_qubits = len(unitary.dims[0])
        for _ in range(samples):
            state = self._random_qubit_state(num_qubits)
            dataset.append((state, unitary * state))
        return dataset

    def feedforward(
        self, samples: Iterable[Tuple[qt.Qobj, qt.Qobj]]
    ) -> List[List[qt.Qobj]]:
        stored_states: List[List[qt.Qobj]] = []
        for sample, _ in samples:
            layerwise = [sample]
            current_state = sample
            for layer in range(1, len(self.arch)):
                current_state = self._layer_channel(layer, current_state)
                layerwise.append(current_state)
            stored_states.append(layerwise)
        return stored_states

    def _layer_channel(self, layer: int, input_state: qt.Qobj) -> qt.Qobj:
        num_in = self.arch[layer - 1]
        num_out = self.arch[layer]
        state = qt.tensor(input_state, self._tensored_zero(num_out))
        layer_unitary = self.unitaries[layer][0].copy()
        for gate in self.unitaries[layer][1:]:
            layer_unitary = gate * layer_unitary
        return self._partial_trace_remove(layer_unitary * state * layer_unitary.dag(), range(num_in))

    @staticmethod
    def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
        return abs((a.dag() * b)[0, 0]) ** 2

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[qt.Qobj], threshold: float,
        *, secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, si), (j, sj) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNNGenQML.state_fidelity(si, sj)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    def sampler_qnn(self) -> qk.QuantumCircuit:
        """Return a simple 2‑qubit parameterised sampler circuit."""
        from qiskit.circuit import ParameterVector
        inputs = ParameterVector("input", 2)
        weights = ParameterVector("weight", 4)
        qc = qk.QuantumCircuit(2)
        qc.ry(inputs[0], 0)
        qc.ry(inputs[1], 1)
        qc.cx(0, 1)
        qc.ry(weights[0], 0)
        qc.ry(weights[1], 1)
        qc.cx(0, 1)
        qc.ry(weights[2], 0)
        qc.ry(weights[3], 1)
        return qc


def SamplerQNN() -> qk.QuantumCircuit:
    """Convenience wrapper to expose the sampler circuit at module level."""
    return GraphQNNGenQML([2, 2]).sampler_qnn()


__all__ = [
    "GraphQNNGenQML",
    "SamplerQNN",
]
