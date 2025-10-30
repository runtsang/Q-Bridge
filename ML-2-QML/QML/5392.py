import numpy as np
import torch
import qiskit
from qiskit import assemble, transpile
import qutip as qt
import networkx as nx
import itertools
from typing import Iterable, List, Tuple, Sequence, Optional

class FCL:
    """Hybrid quantum‑classical layer that unifies a fully‑connected expectation,
    a graph‑based state adjacency, and a sampler QNN."""

    def __init__(self, n_qubits: int = 1, backend=None, shots: int = 100):
        self.n_qubits = n_qubits
        if backend is None:
            backend = qiskit.Aer.get_backend("qasm_simulator")
        self.backend = backend
        self.shots = shots
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> qiskit.QuantumCircuit:
        qc = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = qiskit.circuit.Parameter("theta")
        qc.h(range(self.n_qubits))
        qc.barrier()
        qc.ry(self.theta, range(self.n_qubits))
        qc.measure_all()
        return qc

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        compiled = transpile(self.circuit, self.backend)
        job = qiskit.execute(
            compiled,
            backend=self.backend,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        result = job.result().get_counts(self.circuit)
        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)
        probabilities = counts / self.shots
        expectation = np.sum(states * probabilities)
        return np.array([expectation])

    # ------------------------------------------------------------------
    #  Graph‑based utilities (mirroring the classical GraphQNN helpers)
    # ------------------------------------------------------------------
    def _random_qubit_unitary(self, num_qubits: int) -> qt.Qobj:
        dim = 2 ** num_qubits
        matrix = np.random.normal(size=(dim, dim)) + 1j * np.random.normal(size=(dim, dim))
        unitary = np.linalg.qr(matrix)[0]
        qobj = qt.Qobj(unitary)
        dims = [2] * num_qubits
        qobj.dims = [dims.copy(), dims.copy()]
        return qobj

    def _random_qubit_state(self, num_qubits: int) -> qt.Qobj:
        dim = 2 ** num_qubits
        amplitudes = np.random.normal(size=(dim, 1)) + 1j * np.random.normal(size=(dim, 1))
        amplitudes /= np.linalg.norm(amplitudes)
        state = qt.Qobj(amplitudes)
        state.dims = [[2] * num_qubits, [1] * num_qubits]
        return state

    def random_training_data(self, unitary: qt.Qobj, samples: int) -> List[Tuple[qt.Qobj, qt.Qobj]]:
        dataset: List[Tuple[qt.Qobj, qt.Qobj]] = []
        num_qubits = len(unitary.dims[0])
        for _ in range(samples):
            state = self._random_qubit_state(num_qubits)
            dataset.append((state, unitary * state))
        return dataset

    def random_network(self, qnn_arch: List[int], samples: int):
        target_unitary = self._random_qubit_unitary(qnn_arch[-1])
        training_data = self.random_training_data(target_unitary, samples)
        unitaries: List[List[qt.Qobj]] = [[]]
        for layer in range(1, len(qnn_arch)):
            num_inputs = qnn_arch[layer - 1]
            num_outputs = qnn_arch[layer]
            layer_ops: List[qt.Qobj] = []
            for output in range(num_outputs):
                op = self._random_qubit_unitary(num_inputs + 1)
                if num_outputs > 1:
                    op = qt.tensor(self._random_qubit_unitary(num_inputs + 1), qt.qeye(2 ** (num_outputs - 1)))
                    op = self._swap_registers(op, num_inputs, num_inputs + output)
                layer_ops.append(op)
            unitaries.append(layer_ops)
        return qnn_arch, unitaries, training_data, target_unitary

    def _swap_registers(self, op: qt.Qobj, source: int, target: int) -> qt.Qobj:
        if source == target:
            return op
        order = list(range(len(op.dims[0])))
        order[source], order[target] = order[target], order[source]
        return op.permute(order)

    def _partial_trace_remove(self, state: qt.Qobj, remove: Sequence[int]) -> qt.Qobj:
        keep = list(range(len(state.dims[0])))
        for index in sorted(remove, reverse=True):
            keep.pop(index)
        return state.ptrace(keep)

    def _layer_channel(
        self,
        qnn_arch: Sequence[int],
        unitaries: Sequence[Sequence[qt.Qobj]],
        layer: int,
        input_state: qt.Qobj,
    ) -> qt.Qobj:
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        state = qt.tensor(input_state, qt.qeye(2 ** num_outputs))
        layer_unitary = unitaries[layer][0].copy()
        for gate in unitaries[layer][1:]:
            layer_unitary = gate * layer_unitary
        return self._partial_trace_remove(layer_unitary * state * layer_unitary.dag(), range(num_inputs))

    def feedforward(
        self,
        qnn_arch: Sequence[int],
        unitaries: Sequence[Sequence[qt.Qobj]],
        samples: Iterable[Tuple[qt.Qobj, qt.Qobj]],
    ) -> List[List[qt.Qobj]]:
        stored_states: List[List[qt.Qobj]] = []
        for sample, _ in samples:
            layerwise = [sample]
            current_state = sample
            for layer in range(1, len(qnn_arch)):
                current_state = self._layer_channel(qnn_arch, unitaries, layer, current_state)
                layerwise.append(current_state)
            stored_states.append(layerwise)
        return stored_states

    # ------------------------------------------------------------------
    #  State‑fidelity helpers
    # ------------------------------------------------------------------
    def state_fidelity(self, a: qt.Qobj, b: qt.Qobj) -> float:
        return abs((a.dag() * b)[0, 0]) ** 2

    def fidelity_adjacency(
        self,
        states: Sequence[qt.Qobj],
        threshold: float,
        *,
        secondary: Optional[float] = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = self.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    # ------------------------------------------------------------------
    #  Sampler QNN
    # ------------------------------------------------------------------
    def sampler_qnn(self) -> qiskit.QuantumCircuit:
        """Return a parameterised sampler circuit suitable for training."""
        inputs2 = qiskit.circuit.ParameterVector("input", 2)
        weights2 = qiskit.circuit.ParameterVector("weight", 4)
        qc2 = qiskit.QuantumCircuit(2)
        qc2.ry(inputs2[0], 0)
        qc2.ry(inputs2[1], 1)
        qc2.cx(0, 1)
        qc2.ry(weights2[0], 0)
        qc2.ry(weights2[1], 1)
        qc2.cx(0, 1)
        qc2.ry(weights2[2], 0)
        qc2.ry(weights2[3], 1)
        qc2.measure_all()
        return qc2

__all__ = [
    "FCL",
]
