"""Graph‑neural‑network hybrid for quantum experiments.

This module implements a ``GraphQNNHybrid`` class that mirrors
the classical implementation above but uses Qiskit and
qutip primitives.  All public methods match the classical
counterpart, enabling plug‑in replacement in hybrid workflows.

Typical usage::

    from GraphQNN__gen317 import GraphQNNHybrid
    gnn = GraphQNNHybrid(mode="quantum")
    arch, unitaries, data, target = gnn.random_network([3, 5, 2], samples=10)
    states = gnn.feedforward(arch, unitaries, data)
    G = gnn.fidelity_adjacency([s[-1] for s in states], threshold=0.8)
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import itertools
import networkx as nx
import qiskit
import qutip as qt
import scipy as sc
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
import qiskit_machine_learning.neural_networks

Tensor = qt.Qobj


class GraphQNNHybrid:
    """Hybrid graph‑neural‑network factory for quantum experiments.

    Parameters
    ----------
    mode : {"classical", "quantum"}
        Must be ``"quantum"`` for this module.  The class is kept
        for API symmetry with the classical module.
    """

    def __init__(self, mode: str = "quantum") -> None:
        self.mode = mode
        if mode not in {"classical", "quantum"}:
            raise ValueError("mode must be 'classical' or 'quantum'")

    # ------------------------------------------------------------------ #
    #  Quantum‑only utilities
    # ------------------------------------------------------------------ #

    @staticmethod
    def _tensored_id(num_qubits: int) -> qt.Qobj:
        I = qt.qeye(2 ** num_qubits)
        dims = [2] * num_qubits
        I.dims = [dims.copy(), dims.copy()]
        return I

    @staticmethod
    def _tensored_zero(num_qubits: int) -> qt.Qobj:
        Z = qt.fock(2 ** num_qubits).proj()
        dims = [2] * num_qubits
        Z.dims = [dims.copy(), dims.copy()]
        return Z

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
        mat = sc.random.normal(size=(dim, dim)) + 1j * sc.random.normal(size=(dim, dim))
        # QR‑decomposition gives a random unitary
        Q, _ = sc.linalg.qr(mat)
        qobj = qt.Qobj(Q)
        dims = [2] * num_qubits
        qobj.dims = [dims.copy(), dims.copy()]
        return qobj

    @staticmethod
    def _random_qubit_state(num_qubits: int) -> qt.Qobj:
        dim = 2 ** num_qubits
        vec = sc.random.normal(size=(dim, 1)) + 1j * sc.random.normal(size=(dim, 1))
        vec /= sc.linalg.norm(vec)
        state = qt.Qobj(vec)
        state.dims = [[2] * num_qubits, [1] * num_qubits]
        return state

    def random_training_data(self, unitary: qt.Qobj, samples: int) -> List[Tuple[qt.Qobj, qt.Qobj]]:
        """Generate input/output pairs for unitary training."""
        data: List[Tuple[qt.Qobj, qt.Qobj]] = []
        n = len(unitary.dims[0])
        for _ in range(samples):
            s = self._random_qubit_state(n)
            data.append((s, unitary * s))
        return data

    def random_network(self, qnn_arch: List[int], samples: int):
        """Create a random layered unitary network."""
        target_unitary = self._random_qubit_unitary(qnn_arch[-1])
        training_data = self.random_training_data(target_unitary, samples)

        unitaries: List[List[qt.Qobj]] = [[]]
        for layer in range(1, len(qnn_arch)):
            n_in = qnn_arch[layer - 1]
            n_out = qnn_arch[layer]
            layer_ops: List[qt.Qobj] = []
            for _ in range(n_out):
                op = self._random_qubit_unitary(n_in + 1)
                if n_out > 1:
                    op = qt.tensor(self._random_qubit_unitary(n_in + 1), self._tensored_id(n_out - 1))
                    op = self._swap_registers(op, n_in, n_in + _)
                layer_ops.append(op)
            unitaries.append(layer_ops)

        return qnn_arch, unitaries, training_data, target_unitary

    # ------------------------------------------------------------------ #
    #  Forward propagation
    # ------------------------------------------------------------------ #

    @staticmethod
    def _layer_channel(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qt.Qobj]],
                      layer: int, input_state: qt.Qobj) -> qt.Qobj:
        n_in = qnn_arch[layer - 1]
        n_out = qnn_arch[layer]
        state = qt.tensor(input_state, GraphQNNHybrid._tensored_zero(n_out))
        layer_unitary = unitaries[layer][0].copy()
        for gate in unitaries[layer][1:]:
            layer_unitary = gate * layer_unitary
        return GraphQNNHybrid._partial_trace_remove(
            layer_unitary * state * layer_unitary.dag(), list(range(n_in))
        )

    @staticmethod
    def _partial_trace_remove(state: qt.Qobj, remove: Sequence[int]) -> qt.Qobj:
        keep = list(range(len(state.dims[0])))
        for idx in sorted(remove, reverse=True):
            keep.pop(idx)
        return state.ptrace(keep)

    def feedforward(self, qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qt.Qobj]],
                    samples: Iterable[Tuple[qt.Qobj, qt.Qobj]]) -> List[List[qt.Qobj]]:
        """Propagate states through the quantum network."""
        all_states: List[List[qt.Qobj]] = []
        for inp, _ in samples:
            layerwise = [inp]
            current = inp
            for layer in range(1, len(qnn_arch)):
                current = self._layer_channel(qnn_arch, unitaries, layer, current)
                layerwise.append(current)
            all_states.append(layerwise)
        return all_states

    # ------------------------------------------------------------------ #
    #  Fidelity utilities (share API with classical version)
    # ------------------------------------------------------------------ #

    @staticmethod
    def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
        """Overlap of two pure states."""
        return abs((a.dag() * b)[0, 0]) ** 2

    @staticmethod
    def fidelity_adjacency(states: Sequence[qt.Qobj], threshold: float,
                           *, secondary: float | None = None,
                           secondary_weight: float = 0.5) -> nx.Graph:
        G = nx.Graph()
        G.add_nodes_from(range(len(states)))
        for (i, si), (j, sj) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNNHybrid.state_fidelity(si, sj)
            if fid >= threshold:
                G.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                G.add_edge(i, j, weight=secondary_weight)
        return G

    # ------------------------------------------------------------------ #
    #  Estimator and classifier wrappers (from reference seeds)
    # ------------------------------------------------------------------ #

    @staticmethod
    def estimator_qnn(num_qubits: int = 1) -> qiskit_machine_learning.neural_networks.EstimatorQNN:
        """Return a Qiskit EstimatorQNN with a simple H‑RY‑RX circuit."""
        from qiskit import QuantumCircuit
        from qiskit.circuit import Parameter
        from qiskit.quantum_info import SparsePauliOp
        from qiskit.primitives import StatevectorEstimator

        params = [Parameter("input1"), Parameter("weight1")]
        qc = QuantumCircuit(num_qubits)
        qc.h(0)
        qc.ry(params[0], 0)
        qc.rx(params[1], 0)

        observable = SparsePauliOp.from_list([("Y" * num_qubits, 1)])
        estimator = StatevectorEstimator()
        return qiskit_machine_learning.neural_networks.EstimatorQNN(
            circuit=qc,
            observables=observable,
            input_params=[params[0]],
            weight_params=[params[1]],
            estimator=estimator,
        )

    @staticmethod
    def classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
        """Return a layered ansatz and associated metadata."""
        encoding = ParameterVector("x", num_qubits)
        weights = ParameterVector("theta", num_qubits * depth)

        qc = QuantumCircuit(num_qubits)
        for qubit, param in zip(range(num_qubits), encoding):
            qc.rx(param, qubit)

        idx = 0
        for _ in range(depth):
            for qubit in range(num_qubits):
                qc.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(num_qubits - 1):
                qc.cz(qubit, qubit + 1)

        observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
        return qc, list(encoding), list(weights), observables

    def __repr__(self) -> str:
        return f"<GraphQNNHybrid mode={self.mode!r}>"


__all__ = ["GraphQNNHybrid"]
