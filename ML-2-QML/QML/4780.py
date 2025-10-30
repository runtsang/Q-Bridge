"""
Quantum‑enhanced implementation of the same three‑mode framework.
The quantum module mirrors the public API of the classical one but
uses qutip for the graph QNN, torchquantum for the LSTM, and
qiskit for the variational classifier.
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import qutip as qt
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Quantum LSTM imports ----
import torchquantum as tq
import torchquantum.functional as tqf


# ------------------------------------------------------------- #
#  Quantum Graph QNN helpers
# ------------------------------------------------------------- #
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


def _random_qubit_unitary(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    matrix = (
        (qt.Qobj(
            (qt.rand_unitary(dim, seed=None)).data.todense()
        ))
    )
    qobj = qt.Qobj(matrix)
    dims = [2] * num_qubits
    qobj.dims = [dims.copy(), dims.copy()]
    return qobj


def _random_qubit_state(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    amplitudes = (qt.rand_unitary(dim, seed=None) @ qt.fock(2 ** num_qubits, 0)).data
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


def random_network(qnn_arch: Sequence[int], samples: int):
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


def _swap_registers(op: qt.Qobj, source: int, target: int) -> qt.Qobj:
    if source == target:
        return op
    order = list(range(len(op.dims[0])))
    order[source], order[target] = order[target], order[source]
    return op.permute(order)


def _partial_trace_remove(state: qt.Qobj, remove: Sequence[int]) -> qt.Qobj:
    keep = list(range(len(state.dims[0])))
    for index in sorted(remove, reverse=True):
        keep.pop(index)
    return state.ptrace(keep)


def _layer_channel(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qt.Qobj]], layer: int, input_state: qt.Qobj) -> qt.Qobj:
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    state = qt.tensor(input_state, _tensored_zero(num_outputs))

    layer_unitary = unitaries[layer][0].copy()
    for gate in unitaries[layer][1:]:
        layer_unitary = gate * layer_unitary

    return _partial_trace_remove(layer_unitary * state * layer_unitary.dag(), range(num_inputs))


def feedforward(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qt.Qobj]], samples: Iterable[Tuple[qt.Qobj, qt.Qobj]]):
    stored_states = []
    for sample, _ in samples:
        layerwise = [sample]
        current_state = sample
        for layer in range(1, len(qnn_arch)):
            current_state = _layer_channel(qnn_arch, unitaries, layer, current_state)
            layerwise.append(current_state)
        stored_states.append(layerwise)
    return stored_states


def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
    return abs((a.dag() * b)[0, 0]) ** 2


def fidelity_adjacency(
    states: Sequence[qt.Qobj], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5
) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(s_i, s_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


# ------------------------------------------------------------- #
#  Quantum LSTM helpers (torchquantum)
# ------------------------------------------------------------- #
class QQuantumLayer(tq.QuantumModule):
    """Simple quantum circuit that outputs a measurement vector."""

    def __init__(self, n_wires: int) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "rx", "wires": [0]},
                {"input_idx": [1], "func": "rx", "wires": [1]},
                {"input_idx": [2], "func": "rx", "wires": [2]},
                {"input_idx": [3], "func": "rx", "wires": [3]},
            ]
        )
        self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
        self.encoder(qdev, x)
        for wire, gate in enumerate(self.params):
            gate(qdev, wires=wire)
        for wire in range(self.n_wires):
            if wire == self.n_wires - 1:
                tqf.cnot(qdev, wires=[wire, 0])
            else:
                tqf.cnot(qdev, wires=[wire, wire + 1])
        return self.measure(qdev)


class QuantumQLSTM(nn.Module):
    """LSTM cell with quantum gates."""

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        self.forget = QQuantumLayer(n_qubits)
        self.input = QQuantumLayer(n_qubits)
        self.update = QQuantumLayer(n_qubits)
        self.output = QQuantumLayer(n_qubits)

        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(self.linear_forget(combined)))
            i = torch.sigmoid(self.input(self.linear_input(combined)))
            g = torch.tanh(self.update(self.linear_update(combined)))
            o = torch.sigmoid(self.output(self.linear_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )


class QuantumLSTMTagger(nn.Module):
    """Sequence tagging model that uses the quantum LSTM."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QuantumQLSTM(embedding_dim, hidden_dim, n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


# ------------------------------------------------------------- #
#  Quantum classifier helper (qiskit)
# ------------------------------------------------------------- #
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """Variational ansatz for a binary classifier."""
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    index = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[index], qubit)
            index += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]
    return circuit, list(encoding), list(weights), observables


# ------------------------------------------------------------- #
#  Unified class
# ------------------------------------------------------------- #
class GraphQNNGen:
    """
    Quantum‑enhanced counterpart of the classical GraphQNNGen.
    The public API and mode selection mirror the classical implementation.
    """

    def __init__(self, mode: str = "graph", **kwargs):
        self.mode = mode.lower()
        if self.mode == "graph":
            self.qnn_arch: Sequence[int] = kwargs.get("qnn_arch", [2, 4, 2])
            self.samples: int = kwargs.get("samples", 100)
            self.arch, self.unitaries, self.training_data, self.target = random_network(
                self.qnn_arch, self.samples
            )
        elif self.mode == "lstm":
            self.embedding_dim: int = kwargs.get("embedding_dim", 10)
            self.hidden_dim: int = kwargs.get("hidden_dim", 20)
            self.vocab_size: int = kwargs.get("vocab_size", 100)
            self.tagset_size: int = kwargs.get("tagset_size", 5)
            self.n_qubits: int = kwargs.get("n_qubits", 0)
            self.lstm_tagger = QuantumLSTMTagger(
                self.embedding_dim,
                self.hidden_dim,
                self.vocab_size,
                self.tagset_size,
                self.n_qubits,
            )
        elif self.mode == "classifier":
            self.num_qubits: int = kwargs.get("num_qubits", 4)
            self.depth: int = kwargs.get("depth", 3)
            self.circuit, _, _, _ = build_classifier_circuit(
                self.num_qubits, self.depth
            )
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    # --------------------- Graph QNN methods --------------------- #
    def graph_feedforward(
        self, samples: Iterable[Tuple[qt.Qobj, qt.Qobj]]
    ) -> List[List[qt.Qobj]]:
        if self.mode!= "graph":
            raise RuntimeError("graph_feedforward is only available in graph mode")
        return feedforward(self.arch, self.unitaries, samples)

    def graph_fidelity_graph(
        self, threshold: float, *, secondary: float | None = None
    ) -> nx.Graph:
        if self.mode!= "graph":
            raise RuntimeError("graph_fidelity_graph is only available in graph mode")
        activations = self.graph_feedforward(self.training_data)
        states = [a[-1] for a in activations]
        return fidelity_adjacency(states, threshold, secondary=secondary)

    # --------------------- LSTM methods --------------------- #
    def lstm_tag(self, sentence: torch.Tensor) -> torch.Tensor:
        if self.mode!= "lstm":
            raise RuntimeError("lstm_tag is only available in lstm mode")
        return self.lstm_tagger(sentence)

    # --------------------- Classifier methods --------------------- #
    def classify(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode!= "classifier":
            raise RuntimeError("classify is only available in classifier mode")
        # In the quantum module the forward pass is performed on a simulator;
        # here we simply return the raw quantum circuit for the user to execute.
        return self.circuit

    # --------------------- Convenience --------------------- #
    def __repr__(self) -> str:
        return f"<GraphQNNGen mode={self.mode}>"


__all__ = [
    "GraphQNNGen",
    "random_network",
    "feedforward",
    "fidelity_adjacency",
    "build_classifier_circuit",
    "QuantumLSTMTagger",
    "QuantumQLSTM",
]
