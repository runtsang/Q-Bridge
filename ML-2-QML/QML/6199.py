"""Quantum‑only module that integrates quantum LSTM and graph‑based QNN.

The implementation uses qutip for state manipulation and
networkx for graph utilities.  All components are fully quantum‑oriented
and can be run on a state‑vector simulator.
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple, Dict

import networkx as nx
import qutip as qt
import scipy as sc

# --------------------------------------------------------------------------- #
# 1. Random data generators (quantum)
# --------------------------------------------------------------------------- #
def _tensored_id(num_qubits: int) -> qt.Qobj:
    identity = qt.qeye(2 ** num_qubits)
    dims = [2] * num_qubits
    identity.dims = [dims.copy(), dims.copy()]
    return identity

def _tensored_zero(num_qubits: int) -> qt.Qobj:
    projector = qt.fock(2 ** num_qubits).proj()
    dims = [2] * num_qubits
    projector.dims = [dims.copy(), dims.copy()]
    return projector

def _swap_registers(op: qt.Qobj, source: int, target: int) -> qt.Qobj:
    if source == target:
        return op
    order = list(range(len(op.dims[0])))
    order[source], order[target] = order[target], order[source]
    return op.permute(order)

def _random_qubit_unitary(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    matrix = sc.random.normal(size=(dim, dim)) + 1j * sc.random.normal(size=(dim, dim))
    unitary = sc.linalg.orth(matrix)
    qobj = qt.Qobj(unitary)
    dims = [2] * num_qubits
    qobj.dims = [dims.copy(), dims.copy()]
    return qobj

def _random_qubit_state(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    amplitudes = sc.random.normal(size=(dim, 1)) + 1j * sc.random.normal(size=(dim, 1))
    amplitudes /= sc.linalg.norm(amplitudes)
    state = qt.Qobj(amplitudes)
    state.dims = [[2] * num_qubits, [1] * num_qubits]
    return state

def random_training_data(unitary: qt.Qobj, samples: int) -> List[Tuple[qt.Qobj, qt.Qobj]]:
    dataset: List[Tuple[qt.Qobj, qt.Qobj]] = []
    num_qubits = len(unitary.dims[0])
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        dataset.append((state, unitary * state))
    return dataset

def random_network(qnn_arch: List[int], samples: int):
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    unitaries: List[List[qt.Qobj]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        layer_ops: List[qt.Qobj] = []
        for output in range(num_outputs):
            op = _random_qubit_unitary(num_inputs + 1)
            if num_outputs > 1:
                op = qt.tensor(_random_qubit_unitary(num_inputs + 1), _tensored_id(num_outputs - 1))
                op = _swap_registers(op, num_inputs, num_inputs + output)
            layer_ops.append(op)
        unitaries.append(layer_ops)

    return qnn_arch, unitaries, training_data, target_unitary

# --------------------------------------------------------------------------- #
# 2. Quantum LSTM
# --------------------------------------------------------------------------- #
class QLSTM:
    """Quantum‑enhanced LSTM cell implemented with qutip state‑vector evolution."""
    class _QLayer:
        def __init__(self, n_qubits: int):
            self.n_qubits = n_qubits
            self.unitary = _random_qubit_unitary(n_qubits)

        def __call__(self, x: qt.Qobj) -> qt.Qobj:
            return self.unitary * x

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Random unitary gates for each LSTM gate
        self.forget_gate = self._QLayer(n_qubits)
        self.input_gate = self._QLayer(n_qubits)
        self.update_gate = self._QLayer(n_qubits)
        self.output_gate = self._QLayer(n_qubits)

    def forward(
        self,
        inputs: List[qt.Qobj],
        states: Tuple[qt.Qobj, qt.Qobj] | None = None,
    ) -> Tuple[List[qt.Qobj], Tuple[qt.Qobj, qt.Qobj]]:
        if states is None:
            hx = _random_qubit_state(self.hidden_dim)
            cx = _random_qubit_state(self.hidden_dim)
        else:
            hx, cx = states

        outputs = []
        for inp in inputs:
            combined = qt.tensor(inp, hx)
            f = self.forget_gate(combined)
            i = self.input_gate(combined)
            g = self.update_gate(combined)
            o = self.output_gate(combined)

            # Simplified recurrence – no nonlinearities for brevity
            cx = f + i + g
            hx = o * qt.tanh(cx)
            outputs.append(hx)
        return outputs, (hx, cx)

# --------------------------------------------------------------------------- #
# 3. Graph‑based QNN (qutip)
# --------------------------------------------------------------------------- #
class GraphQNN:
    """Graph‑based quantum neural network using qutip unitaries."""
    def __init__(self, qnn_arch: List[int], unitaries: List[List[qt.Qobj]]):
        self.arch = qnn_arch
        self.unitaries = unitaries

    def feedforward(self, samples: Iterable[Tuple[qt.Qobj, qt.Qobj]]):
        stored_states = []
        for sample, _ in samples:
            current = sample
            layerwise = [current]
            for layer in range(1, len(self.arch)):
                ops = self.unitaries[layer]
                for op in ops:
                    current = op * current
                layerwise.append(current)
            stored_states.append(layerwise)
        return stored_states

    def fidelity(self, a: qt.Qobj, b: qt.Qobj) -> float:
        return abs((a.dag() * b)[0, 0]) ** 2

    def fidelity_adjacency(
        self,
        states: Sequence[qt.Qobj],
        threshold: float,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
            fid = self.fidelity(s_i, s_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

# --------------------------------------------------------------------------- #
# 4. Unified wrapper
# --------------------------------------------------------------------------- #
class UnifiedQLSTMGraphQNN:
    """Composite quantum model coupling a quantum LSTM tagger and a graph‑QNN."""
    def __init__(self, lstm_config: Dict, graph_config: Dict):
        self.tagger = QLSTM(**lstm_config)
        self.graph = GraphQNN(**graph_config)

    def tag_sequence(self, sentence: List[qt.Qobj]) -> List[qt.Qobj]:
        return self.tagger.forward(sentence)

    def propagate_graph(
        self, samples: Iterable[Tuple[qt.Qobj, qt.Qobj]]
    ):
        return self.graph.feedforward(samples)

    def graph_fidelity_adjacency(
        self,
        states: Sequence[qt.Qobj],
        threshold: float,
        secondary: float | None = None,
    ) -> nx.Graph:
        return self.graph.fidelity_adjacency(states, threshold, secondary=secondary)

__all__ = ["QLSTM", "GraphQNN", "UnifiedQLSTMGraphQNN"]
