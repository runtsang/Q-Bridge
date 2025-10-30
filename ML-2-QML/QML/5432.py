"""Combined quantum graph neural network and estimator utilities.

The class ``GraphQNNGen325`` mirrors the classical interface but uses
Qiskit and QuTiP primitives.  It can build variational circuits,
evaluate state‑vector fidelities, and construct a weighted adjacency
graph from the resulting states.  A minimal estimator that returns the
expectation value of a list of Pauli operators is also provided.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import Callable, List, Tuple

import numpy as np
import networkx as nx
import qiskit
import qutip as qt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator

__all__ = ["GraphQNNGen325"]


class GraphQNNGen325:
    """Unified quantum interface for graph‑based neural nets and estimators."""

    # ------------------------------------------------------------------
    #  Utility helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _tensored_id(num_qubits: int) -> qt.Qobj:
        """Return a tensor‑product identity with proper dim tags."""
        I = qt.qeye(2 ** num_qubits)
        dims = [2] * num_qubits
        I.dims = [dims.copy(), dims.copy()]
        return I

    @staticmethod
    def _tensored_zero(num_qubits: int) -> qt.Qobj:
        """Return a projector on the zero state with dimensional tags."""
        proj = qt.fock(2 ** num_qubits).proj()
        dims = [2] * num_qubits
        proj.dims = [dims.copy(), dims.copy()]
        return proj

    @staticmethod
    def _random_qubit_unitary(num_qubits: int) -> qt.Qobj:
        """Generate a Haar‑random unitary as a Qobj."""
        dim = 2 ** num_qubits
        mat = np.random.normal(size=(dim, dim)) + 1j * np.random.normal(size=(dim, dim))
        u, _ = np.linalg.qr(mat)
        q = qt.Qobj(u)
        dims = [2] * num_qubits
        q.dims = [dims.copy(), dims.copy()]
        return q

    @staticmethod
    def _random_qubit_state(num_qubits: int) -> qt.Qobj:
        """Sample a random pure state on ``num_qubits`` qubits."""
        dim = 2 ** num_qubits
        vec = np.random.normal(size=(dim, 1)) + 1j * np.random.normal(size=(dim, 1))
        vec /= np.linalg.norm(vec)
        s = qt.Qobj(vec)
        s.dims = [[2] * num_qubits, [1] * num_qubits]
        return s

    @staticmethod
    def _swap_registers(op: qt.Qobj, src: int, tgt: int) -> qt.Qobj:
        if src == tgt:
            return op
        order = list(range(len(op.dims[0])))
        order[src], order[tgt] = order[tgt], order[src]
        return op.permute(order)

    # ------------------------------------------------------------------
    #  Random training data and network
    # ------------------------------------------------------------------
    @staticmethod
    def random_training_data(unitary: qt.Qobj, samples: int) -> List[Tuple[qt.Qobj, qt.Qobj]]:
        """Generate input–output pairs by applying a fixed unitary."""
        data: List[Tuple[qt.Qobj, qt.Qobj]] = []
        nq = len(unitary.dims[0])
        for _ in range(samples):
            s = GraphQNNGen325._random_qubit_state(nq)
            data.append((s, unitary * s))
        return data

    @staticmethod
    def random_network(qnn_arch: List[int], samples: int):
        """Build a variational circuit stack and corresponding training data."""
        target = GraphQNNGen325._random_qubit_unitary(qnn_arch[-1])
        training = GraphQNNGen325.random_training_data(target, samples)

        ops: List[List[qt.Qobj]] = [[]]
        for layer in range(1, len(qnn_arch)):
            in_n = qnn_arch[layer - 1]
            out_n = qnn_arch[layer]
            layer_ops: List[qt.Qobj] = []
            for out_idx in range(out_n):
                op = GraphQNNGen325._random_qubit_unitary(in_n + 1)
                if out_n > 1:
                    op = qt.tensor(op, GraphQNNGen325._tensored_id(out_n - 1))
                    op = GraphQNNGen325._swap_registers(op, in_n, in_n + out_idx)
                layer_ops.append(op)
            ops.append(layer_ops)
        return qnn_arch, ops, training, target

    # ------------------------------------------------------------------
    #  Forward pass
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
        return GraphQNNGen325._partial_trace_keep(state, keep)

    @staticmethod
    def _layer_channel(qnn_arch: Sequence[int], ops: Sequence[Sequence[qt.Qobj]], layer: int, input_state: qt.Qobj) -> qt.Qobj:
        in_n = qnn_arch[layer - 1]
        out_n = qnn_arch[layer]
        state = qt.tensor(input_state, GraphQNNGen325._tensored_zero(out_n))
        u = ops[layer][0].copy()
        for gate in ops[layer][1:]:
            u = gate * u
        return GraphQNNGen325._partial_trace_remove(u * state * u.dag(), range(in_n))

    @staticmethod
    def feedforward(qnn_arch: Sequence[int], ops: Sequence[Sequence[qt.Qobj]], samples: Iterable[Tuple[qt.Qobj, qt.Qobj]]) -> List[List[qt.Qobj]]:
        """Propagate each input state through all layers."""
        all_states: List[List[qt.Qobj]] = []
        for inp, _ in samples:
            layerwise: List[qt.Qobj] = [inp]
            cur = inp
            for l in range(1, len(qnn_arch)):
                cur = GraphQNNGen325._layer_channel(qnn_arch, ops, l, cur)
                layerwise.append(cur)
            all_states.append(layerwise)
        return all_states

    # ------------------------------------------------------------------
    #  Fidelity and adjacency
    # ------------------------------------------------------------------
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
        G = nx.Graph()
        G.add_nodes_from(range(len(states)))
        for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNNGen325.state_fidelity(a, b)
            if fid >= threshold:
                G.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                G.add_edge(i, j, weight=secondary_weight)
        return G

    # ------------------------------------------------------------------
    #  Quantum classifier circuit
    # ------------------------------------------------------------------
    @staticmethod
    def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
        """Return a layered ansatz with explicit encoding."""
        enc = qiskit.circuit.ParameterVector("x", num_qubits)
        wts = qiskit.circuit.ParameterVector("theta", num_qubits * depth)
        qc = QuantumCircuit(num_qubits)
        for qubit, param in zip(range(num_qubits), enc):
            qc.rx(param, qubit)
        idx = 0
        for _ in range(depth):
            for qubit in range(num_qubits):
                qc.ry(wts[idx], qubit)
                idx += 1
            for qubit in range(num_qubits - 1):
                qc.cz(qubit, qubit + 1)
        obs = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
        return qc, list(enc), list(wts), obs

    # ------------------------------------------------------------------
    #  Quantum self‑attention
    # ------------------------------------------------------------------
    @staticmethod
    def SelfAttention(n_qubits: int = 4):
        """Return a minimal quantum self‑attention circuit."""

        class QuantumSelfAttention:
            def __init__(self, n_qubits: int = n_qubits):
                self.n_qubits = n_qubits
                self.qr = QuantumRegister(n_qubits, "q")
                self.cr = ClassicalRegister(n_qubits, "c")

            def _build(self, rot: np.ndarray, ent: np.ndarray) -> QuantumCircuit:
                qc = QuantumCircuit(self.qr, self.cr)
                for i in range(self.n_qubits):
                    qc.rx(rot[3 * i], i)
                    qc.ry(rot[3 * i + 1], i)
                    qc.rz(rot[3 * i + 2], i)
                for i in range(self.n_qubits - 1):
                    qc.crx(ent[i], i, i + 1)
                qc.measure(self.qr, self.cr)
                return qc

            def run(
                self,
                backend,
                rot: np.ndarray,
                ent: np.ndarray,
                shots: int = 1024,
            ):
                qc = self._build(rot, ent)
                job = execute(qc, backend, shots=shots)
                return job.result().get_counts(qc)

        backend = Aer.get_backend("qasm_simulator")
        return QuantumSelfAttention()

    # ------------------------------------------------------------------
    #  Fast estimator
    # ------------------------------------------------------------------
    class FastBaseEstimator:
        """Return expectation values for a parametrised circuit."""

        def __init__(self, circuit: QuantumCircuit):
            self._circuit = circuit
            self._params = list(circuit.parameters)

        def _bind(self, values: Sequence[float]) -> QuantumCircuit:
            if len(values)!= len(self._params):
                raise ValueError("Parameter count mismatch.")
            mapping = dict(zip(self._params, values))
            return self._circuit.assign_parameters(mapping, inplace=False)

        def evaluate(self, observables: Iterable[BaseOperator], parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
            observables = list(observables)
            results: List[List[complex]] = []
            for vals in parameter_sets:
                st = Statevector.from_instruction(self._bind(vals))
                row = [st.expectation_value(obs) for obs in observables]
                results.append(row)
            return results
