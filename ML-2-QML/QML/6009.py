"""Combined quantum graph neural network and variational classifier.

Implements a quantum‑style feed‑forward network built from unitary layers
and a variational ansatz for classification.  The interface mirrors the
classical side so that the same experiment can be reproduced on a
quantum simulator.

Key components
* Random unitary generation and training data.
* Layer‑wise state propagation with partial tracing.
* Fidelity‑based adjacency graph.
* A Qiskit circuit factory for a shallow variational classifier.
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import qutip as qt
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
import scipy as sc

Qobj = qt.Qobj


class GraphQNNClassifier:
    """Dual‑mode Graph‑QNN classifier (quantum implementation)."""

    def __init__(self, qnn_arch: Sequence[int], depth: int, num_qubits: int):
        self.qnn_arch = list(qnn_arch)
        self.depth = depth
        self.num_qubits = num_qubits

    # --------------------------------------------------------------------- #
    #  Quantum utilities
    # --------------------------------------------------------------------- #

    @staticmethod
    def _tensored_id(num_qubits: int) -> Qobj:
        I = qt.qeye(2 ** num_qubits)
        dims = [2] * num_qubits
        I.dims = [dims.copy(), dims.copy()]
        return I

    @staticmethod
    def _tensored_zero(num_qubits: int) -> Qobj:
        zero = qt.fock(2 ** num_qubits).proj()
        dims = [2] * num_qubits
        zero.dims = [dims.copy(), dims.copy()]
        return zero

    @staticmethod
    def _swap_registers(op: Qobj, source: int, target: int) -> Qobj:
        if source == target:
            return op
        order = list(range(len(op.dims[0])))
        order[source], order[target] = order[target], order[source]
        return op.permute(order)

    @staticmethod
    def _random_qubit_unitary(num_qubits: int) -> Qobj:
        dim = 2 ** num_qubits
        mat = sc.random.normal(size=(dim, dim)) + 1j * sc.random.normal(size=(dim, dim))
        U = sc.linalg.orth(mat)
        qobj = qt.Qobj(U)
        dims = [2] * num_qubits
        qobj.dims = [dims.copy(), dims.copy()]
        return qobj

    @staticmethod
    def _random_qubit_state(num_qubits: int) -> Qobj:
        dim = 2 ** num_qubits
        vec = sc.random.normal(size=(dim, 1)) + 1j * sc.random.normal(size=(dim, 1))
        vec /= sc.linalg.norm(vec)
        state = qt.Qobj(vec)
        state.dims = [[2] * num_qubits, [1] * num_qubits]
        return state

    def random_network(self, samples: int) -> Tuple[List[int], List[List[Qobj]], List[Tuple[Qobj, Qobj]], Qobj]:
        """Generate random unitary layers and training data for the final layer."""
        target_unitary = self._random_qubit_unitary(self.qnn_arch[-1])
        training_data = self.random_training_data(target_unitary, samples)

        unitaries: List[List[Qobj]] = [[]]
        for layer in range(1, len(self.qnn_arch)):
            in_q = self.qnn_arch[layer - 1]
            out_q = self.qnn_arch[layer]
            layer_ops: List[Qobj] = []
            for output in range(out_q):
                op = self._random_qubit_unitary(in_q + 1)
                if out_q > 1:
                    op = qt.tensor(self._random_qubit_unitary(in_q + 1), self._tensored_id(out_q - 1))
                    op = self._swap_registers(op, in_q, in_q + output)
                layer_ops.append(op)
            unitaries.append(layer_ops)

        return self.qnn_arch, unitaries, training_data, target_unitary

    @staticmethod
    def random_training_data(unitary: Qobj, samples: int) -> List[Tuple[Qobj, Qobj]]:
        """Generate (state, U|state>) pairs for a target unitary."""
        data: List[Tuple[Qobj, Qobj]] = []
        num_q = len(unitary.dims[0])
        for _ in range(samples):
            state = GraphQNNClassifier._random_qubit_state(num_q)
            data.append((state, unitary * state))
        return data

    def _partial_trace_keep(self, state: Qobj, keep: Sequence[int]) -> Qobj:
        if len(keep)!= len(state.dims[0]):
            return state.ptrace(list(keep))
        return state

    def _partial_trace_remove(self, state: Qobj, remove: Sequence[int]) -> Qobj:
        keep = list(range(len(state.dims[0])))
        for idx in sorted(remove, reverse=True):
            keep.pop(idx)
        return self._partial_trace_keep(state, keep)

    def _layer_channel(
        self,
        unitaries: Sequence[Sequence[Qobj]],
        layer: int,
        input_state: Qobj,
    ) -> Qobj:
        in_q = self.qnn_arch[layer - 1]
        out_q = self.qnn_arch[layer]
        state = qt.tensor(input_state, self._tensored_zero(out_q))

        layer_unitary = unitaries[layer][0].copy()
        for gate in unitaries[layer][1:]:
            layer_unitary = gate * layer_unitary

        return self._partial_trace_remove(layer_unitary * state * layer_unitary.dag(), range(in_q))

    def feedforward(
        self,
        unitaries: Sequence[Sequence[Qobj]],
        samples: Iterable[Tuple[Qobj, Qobj]],
    ) -> List[List[Qobj]]:
        """Propagate each input state through all layers."""
        all_states: List[List[Qobj]] = []
        for sample, _ in samples:
            layerwise: List[Qobj] = [sample]
            state = sample
            for layer in range(1, len(self.qnn_arch)):
                state = self._layer_channel(unitaries, layer, state)
                layerwise.append(state)
            all_states.append(layerwise)
        return all_states

    @staticmethod
    def state_fidelity(a: Qobj, b: Qobj) -> float:
        """Squared overlap of two pure quantum states."""
        return abs((a.dag() * b)[0, 0]) ** 2

    def fidelity_adjacency(
        self,
        states: Sequence[Qobj],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Build a graph where edges encode quantum state fidelity."""
        G = nx.Graph()
        G.add_nodes_from(range(len(states)))
        for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
            fid = self.state_fidelity(a, b)
            if fid >= threshold:
                G.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                G.add_edge(i, j, weight=secondary_weight)
        return G

    # --------------------------------------------------------------------- #
    #  Variational classifier factory
    # --------------------------------------------------------------------- #

    def build_classifier(self) -> Tuple[QuantumCircuit, List, List, List[SparsePauliOp]]:
        """Return a Qiskit variational circuit and metadata."""
        encoding = ParameterVector("x", self.num_qubits)
        weights = ParameterVector("theta", self.num_qubits * self.depth)

        qc = QuantumCircuit(self.num_qubits)
        for param, qubit in zip(encoding, range(self.num_qubits)):
            qc.rx(param, qubit)

        idx = 0
        for _ in range(self.depth):
            for qubit in range(self.num_qubits):
                qc.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(self.num_qubits - 1):
                qc.cz(qubit, qubit + 1)

        observables = [SparsePauliOp("I" * i + "Z" + "I" * (self.num_qubits - i - 1))
                       for i in range(self.num_qubits)]
        return qc, list(encoding), list(weights), observables
