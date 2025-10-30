from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
import networkx as nx
import qutip as qt
import scipy as sc
import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.random import random_circuit


class GraphQNNGen099QML:
    """Quantum graph neural network utilities with quanvolution and variational classifier support."""

    @staticmethod
    def _tensored_id(num_qubits: int) -> qt.Qobj:
        identity = qt.qeye(2 ** num_qubits)
        dims = [2] * num_qubits
        identity.dims = [dims.copy(), dims.copy()]
        return identity

    @staticmethod
    def _tensored_zero(num_qubits: int) -> qt.Qobj:
        projector = qt.fock(2 ** num_qubits).proj()
        dims = [2] * num_qubits
        projector.dims = [dims.copy(), dims.copy()]
        return projector

    @staticmethod
    def _swap_registers(op: qt.Qobj, source: int, target: int) -> qt.Qobj:
        if source == target:
            return op
        order = list(range(len(op.dims[0])))
        order[source], order[target] = order[target], order[source]
        return op.permute(order)

    @staticmethod
    def _random_qubit_unitary(num_qubits: int) -> qt.Qobj:
        dim = 2 ** num_qubits
        matrix = (
            sc.random.normal(size=(dim, dim))
            + 1j * sc.random.normal(size=(dim, dim))
        )
        unitary = sc.linalg.orth(matrix)
        qobj = qt.Qobj(unitary)
        dims = [2] * num_qubits
        qobj.dims = [dims.copy(), dims.copy()]
        return qobj

    @staticmethod
    def _random_qubit_state(num_qubits: int) -> qt.Qobj:
        dim = 2 ** num_qubits
        amplitudes = (
            sc.random.normal(size=(dim, 1))
            + 1j * sc.random.normal(size=(dim, 1))
        )
        amplitudes /= sc.linalg.norm(amplitudes)
        state = qt.Qobj(amplitudes)
        state.dims = [[2] * num_qubits, [1] * num_qubits]
        return state

    @staticmethod
    def random_training_data(unitary: qt.Qobj, samples: int) -> list[tuple[qt.Qobj, qt.Qobj]]:
        dataset: list[tuple[qt.Qobj, qt.Qobj]] = []
        num_qubits = len(unitary.dims[0])
        for _ in range(samples):
            state = GraphQNNGen099QML._random_qubit_state(num_qubits)
            dataset.append((state, unitary * state))
        return dataset

    @staticmethod
    def random_network(qnn_arch: list[int], samples: int):
        target_unitary = GraphQNNGen099QML._random_qubit_unitary(qnn_arch[-1])
        training_data = GraphQNNGen099QML.random_training_data(target_unitary, samples)

        unitaries: list[list[qt.Qobj]] = [[]]
        for layer in range(1, len(qnn_arch)):
            num_inputs = qnn_arch[layer - 1]
            num_outputs = qnn_arch[layer]
            layer_ops: list[qt.Qobj] = []
            for output in range(num_outputs):
                op = GraphQNNGen099QML._random_qubit_unitary(num_inputs + 1)
                if num_outputs > 1:
                    op = qt.tensor(
                        GraphQNNGen099QML._random_qubit_unitary(num_inputs + 1),
                        GraphQNNGen099QML._tensored_id(num_outputs - 1),
                    )
                    op = GraphQNNGen099QML._swap_registers(op, num_inputs, num_inputs + output)
                layer_ops.append(op)
            unitaries.append(layer_ops)

        return qnn_arch, unitaries, training_data, target_unitary

    @staticmethod
    def _partial_trace_keep(state: qt.Qobj, keep: Sequence[int]) -> qt.Qobj:
        if len(keep)!= len(state.dims[0]):
            return state.ptrace(list(keep))
        return state

    @staticmethod
    def _partial_trace_remove(state: qt.Qobj, remove: Sequence[int]) -> qt.Qobj:
        keep = list(range(len(state.dims[0])))
        for index in sorted(remove, reverse=True):
            keep.pop(index)
        return GraphQNNGen099QML._partial_trace_keep(state, keep)

    @staticmethod
    def _layer_channel(
        qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qt.Qobj]], layer: int, input_state: qt.Qobj
    ) -> qt.Qobj:
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        state = qt.tensor(input_state, GraphQNNGen099QML._tensored_zero(num_outputs))

        layer_unitary = unitaries[layer][0].copy()
        for gate in unitaries[layer][1:]:
            layer_unitary = gate * layer_unitary

        return GraphQNNGen099QML._partial_trace_remove(
            layer_unitary * state * layer_unitary.dag(), range(num_inputs)
        )

    @staticmethod
    def feedforward(
        qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qt.Qobj]], samples: Iterable[tuple[qt.Qobj, qt.Qobj]]
    ):
        stored_states: list[list[qt.Qobj]] = []
        for sample, _ in samples:
            layerwise = [sample]
            current_state = sample
            for layer in range(1, len(qnn_arch)):
                current_state = GraphQNNGen099QML._layer_channel(
                    qnn_arch, unitaries, layer, current_state
                )
                layerwise.append(current_state)
            stored_states.append(layerwise)
        return stored_states

    @staticmethod
    def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
        """Return the absolute squared overlap between pure states a and b."""
        return abs((a.dag() * b)[0, 0]) ** 2

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[qt.Qobj], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5
    ) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNNGen099QML.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    @staticmethod
    def Conv(
        kernel_size: int = 2, backend=None, shots: int = 100, threshold: float = 127
    ) -> qiskit.QuantumCircuit:
        """Return a quantum quanvolution filter circuit."""
        class QuanvCircuit:
            def __init__(self, kernel_size, backend, shots, threshold):
                self.n_qubits = kernel_size ** 2
                self._circuit = QuantumCircuit(self.n_qubits)
                self.theta = [
                    qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)
                ]
                for i in range(self.n_qubits):
                    self._circuit.rx(self.theta[i], i)
                self._circuit.barrier()
                self._circuit += random_circuit(self.n_qubits, 2)
                self._circuit.measure_all()

                self.backend = backend
                self.shots = shots
                self.threshold = threshold

            def run(self, data):
                data = np.reshape(data, (1, self.n_qubits))
                param_binds = []
                for dat in data:
                    bind = {}
                    for i, val in enumerate(dat):
                        bind[self.theta[i]] = np.pi if val > self.threshold else 0
                    param_binds.append(bind)

                job = qiskit.execute(
                    self._circuit,
                    self.backend,
                    shots=self.shots,
                    parameter_binds=param_binds,
                )
                result = job.result().get_counts(self._circuit)

                counts = 0
                for key, val in result.items():
                    ones = sum(int(bit) for bit in key)
                    counts += ones * val

                return counts / (self.shots * self.n_qubits)

        if backend is None:
            backend = qiskit.Aer.get_backend("qasm_simulator")
        return QuanvCircuit(kernel_size, backend, shots, threshold)

    @staticmethod
    def build_classifier_circuit(num_qubits: int, depth: int):
        encoding = ParameterVector("x", num_qubits)
        weights = ParameterVector("theta", num_qubits * depth)

        circuit = QuantumCircuit(num_qubits)
        for param, qubit in zip(encoding, range(num_qubits)):
            circuit.rx(param, qubit)

        idx = 0
        for _ in range(depth):
            for qubit in range(num_qubits):
                circuit.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(num_qubits - 1):
                circuit.cz(qubit, qubit + 1)

        observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)
        ]
        return circuit, list(encoding), list(weights), observables


__all__ = [
    "GraphQNNGen099QML",
]
