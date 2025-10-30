"""UnifiedQuantumGraphLayer – quantum implementation.

The class provides a parameterized Qiskit circuit, Qutip‑based state
propagation, and graph utilities identical to the classical version.
It exposes the same public API (`run`, `feedforward`, `state_fidelity`,
`fidelity_adjacency`, `random_network`, `random_training_data`) so
experiments can switch between classical and quantum backends
without changing client code.
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
import qiskit
import qutip as qt
import scipy as sc

__all__ = ["UnifiedQuantumGraphLayer"]


class UnifiedQuantumGraphLayer:
    """Hybrid quantum graph layer.

    Implements a parameterized Qiskit circuit for a single‑qubit
    fully‑connected layer and a stack of Qutip unitary layers
    for graph‑based quantum neural networks.  The API mirrors the
    classical variant to enable side‑by‑side experimentation.
    """

    def __init__(self, n_qubits: int = 1, backend=None, shots: int = 100) -> None:
        self.n_qubits = n_qubits
        if backend is None:
            backend = qiskit.Aer.get_backend("qasm_simulator")
        self.backend = backend
        self.shots = shots
        self._build_circuit()

    def _build_circuit(self) -> None:
        """Construct a simple parameterized circuit."""
        self.circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = qiskit.circuit.Parameter("theta")
        self.circuit.h(range(self.n_qubits))
        self.circuit.barrier()
        self.circuit.ry(self.theta, range(self.n_qubits))
        self.circuit.measure_all()

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Execute the circuit for each theta and return expectation values."""
        job = qiskit.execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        result = job.result().get_counts(self.circuit)
        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)
        probs = counts / self.shots
        expectation = np.sum(states * probs)
        return np.array([expectation])

    # ------------------------------------------------------------------
    # Qutip helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _tensored_id(num_qubits: int) -> qt.Qobj:
        return qt.qeye(2 ** num_qubits)

    @staticmethod
    def _tensored_zero(num_qubits: int) -> qt.Qobj:
        return qt.fock(2 ** num_qubits).proj()

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
        matrix = sc.random.normal(size=(dim, dim)) + 1j * sc.random.normal(size=(dim, dim))
        unitary = sc.linalg.orth(matrix)
        qobj = qt.Qobj(unitary)
        qobj.dims = [[2] * num_qubits, [2] * num_qubits]
        return qobj

    @staticmethod
    def _random_qubit_state(num_qubits: int) -> qt.Qobj:
        dim = 2 ** num_qubits
        amps = sc.random.normal(size=(dim, 1)) + 1j * sc.random.normal(size=(dim, 1))
        amps /= sc.linalg.norm(amps)
        state = qt.Qobj(amps)
        state.dims = [[2] * num_qubits, [1] * num_qubits]
        return state

    # ------------------------------------------------------------------
    # Training data generation
    # ------------------------------------------------------------------
    @staticmethod
    def random_training_data(unitary: qt.Qobj, samples: int) -> List[Tuple[qt.Qobj, qt.Qobj]]:
        dataset: List[Tuple[qt.Qobj, qt.Qobj]] = []
        num_qubits = len(unitary.dims[0])
        for _ in range(samples):
            state = UnifiedQuantumGraphLayer._random_qubit_state(num_qubits)
            dataset.append((state, unitary * state))
        return dataset

    # ------------------------------------------------------------------
    # Random network construction
    # ------------------------------------------------------------------
    @staticmethod
    def random_network(qnn_arch: List[int], samples: int):
        target_unitary = UnifiedQuantumGraphLayer._random_qubit_unitary(qnn_arch[-1])
        training_data = UnifiedQuantumGraphLayer.random_training_data(target_unitary, samples)

        unitaries: List[List[qt.Qobj]] = [[]]
        for layer in range(1, len(qnn_arch)):
            num_inputs = qnn_arch[layer - 1]
            num_outputs = qnn_arch[layer]
            layer_ops: List[qt.Qobj] = []
            for output in range(num_outputs):
                op = UnifiedQuantumGraphLayer._random_qubit_unitary(num_inputs + 1)
                if num_outputs > 1:
                    op = qt.tensor(
                        UnifiedQuantumGraphLayer._random_qubit_unitary(num_inputs + 1),
                        UnifiedQuantumGraphLayer._tensored_id(num_outputs - 1),
                    )
                    op = UnifiedQuantumGraphLayer._swap_registers(
                        op, num_inputs, num_inputs + output
                    )
                layer_ops.append(op)
            unitaries.append(layer_ops)

        return qnn_arch, unitaries, training_data, target_unitary

    # ------------------------------------------------------------------
    # Partial trace helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _partial_trace_keep(state: qt.Qobj, keep: Sequence[int]) -> qt.Qobj:
        if len(keep)!= len(state.dims[0]):
            return state.ptrace(list(keep))
        return state

    @staticmethod
    def _partial_trace_remove(state: qt.Qobj, remove: Sequence[int]) -> qt.Qobj:
        keep = list(range(len(state.dims[0])))
        for idx in sorted(remove, reverse=True):
            keep.pop(idx)
        return UnifiedQuantumGraphLayer._partial_trace_keep(state, keep)

    # ------------------------------------------------------------------
    # Layer channel
    # ------------------------------------------------------------------
    @staticmethod
    def _layer_channel(
        qnn_arch: Sequence[int],
        unitaries: Sequence[Sequence[qt.Qobj]],
        layer: int,
        input_state: qt.Qobj,
    ) -> qt.Qobj:
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        state = qt.tensor(input_state, UnifiedQuantumGraphLayer._tensored_zero(num_outputs))
        layer_unitary = unitaries[layer][0].copy()
        for gate in unitaries[layer][1:]:
            layer_unitary = gate * layer_unitary
        return UnifiedQuantumGraphLayer._partial_trace_remove(
            layer_unitary * state * layer_unitary.dag(),
            range(num_inputs),
        )

    # ------------------------------------------------------------------
    # Feed‑forward propagation
    # ------------------------------------------------------------------
    @staticmethod
    def feedforward(
        qnn_arch: Sequence[int],
        unitaries: Sequence[Sequence[qt.Qobj]],
        samples: Iterable[Tuple[qt.Qobj, qt.Qobj]],
    ) -> List[List[qt.Qobj]]:
        stored_states: List[List[qt.Qobj]] = []
        for sample, _ in samples:
            layerwise: List[qt.Qobj] = [sample]
            current_state = sample
            for layer in range(1, len(qnn_arch)):
                current_state = UnifiedQuantumGraphLayer._layer_channel(
                    qnn_arch, unitaries, layer, current_state
                )
                layerwise.append(current_state)
            stored_states.append(layerwise)
        return stored_states

    # ------------------------------------------------------------------
    # Graph utilities
    # ------------------------------------------------------------------
    @staticmethod
    def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
        """Absolute squared overlap between pure states."""
        return abs((a.dag() * b)[0, 0]) ** 2

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[qt.Qobj],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Build a weighted adjacency graph from state fidelities."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = UnifiedQuantumGraphLayer.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph
