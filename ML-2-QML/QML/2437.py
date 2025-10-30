"""
Quantum‑enhanced LSTM and graph utilities.

This module implements:
* QLSTM: an LSTM cell whose gates are realised by a small variational quantum circuit.
* LSTMTagger: same as the classical counterpart but using the quantum LSTM.
* GraphQNN: feed‑forward quantum neural network utilities that operate on pure states.
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import qutip as qt
import scipy as sc
import torch
import torch.nn as nn
import torch.quantum as tq
import torch.quantum.functional as tqf


# --------------------------------------------------------------------------- #
# Quantum LSTM
# --------------------------------------------------------------------------- #

class QLSTM(nn.Module):
    """Quantum‑enhanced LSTM cell with variational gates."""

    class QLayer(tq.QuantumModule):
        """Small variational circuit that processes a vector of qubit‑encoded inputs."""

        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [i], "func": "rx", "wires": [i]}
                    for i in range(min(4, n_wires))
                ]
            )
            self.params = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            for wire in range(self.n_wires):
                tgt = 0 if wire == self.n_wires - 1 else wire + 1
                tqf.cnot(qdev, wires=[wire, tgt])
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Quantum layers for each gate
        self.forget = self.QLayer(n_qubits)
        self.input = self.QLayer(n_qubits)
        self.update = self.QLayer(n_qubits)
        self.output = self.QLayer(n_qubits)

        # Linear projections to the quantum space
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

        # Linear mapping back to hidden dimension
        self.map_forget = nn.Linear(n_qubits, hidden_dim)
        self.map_input = nn.Linear(n_qubits, hidden_dim)
        self.map_update = nn.Linear(n_qubits, hidden_dim)
        self.map_output = nn.Linear(n_qubits, hidden_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []

        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)

            f_q = self.linear_forget(combined)
            i_q = self.linear_input(combined)
            g_q = self.linear_update(combined)
            o_q = self.linear_output(combined)

            f = torch.sigmoid(self.map_forget(self.forget(f_q)))
            i = torch.sigmoid(self.map_input(self.input(i_q)))
            g = torch.tanh(self.map_update(self.update(g_q)))
            o = torch.sigmoid(self.map_output(self.output(o_q)))

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))

        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx


class LSTMTagger(nn.Module):
    """Sequence tagging model that switches between classical and quantum LSTM."""

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
        self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits) if n_qubits > 0 else nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


# --------------------------------------------------------------------------- #
# Quantum graph utilities (mirroring GraphQNN)
# --------------------------------------------------------------------------- #

def _tensored_id(num_qubits: int) -> qt.Qobj:
    """Identity operator for a register of `num_qubits` qubits."""
    identity = qt.qeye(2 ** num_qubits)
    dims = [2] * num_qubits
    identity.dims = [dims.copy(), dims.copy()]
    return identity


def _tensored_zero(num_qubits: int) -> qt.Qobj:
    """Projector onto the all‑zero state for `num_qubits` qubits."""
    projector = qt.fock(2 ** num_qubits).proj()
    dims = [2] * num_qubits
    projector.dims = [dims.copy(), dims.copy()]
    return projector


def _swap_registers(op: qt.Qobj, source: int, target: int) -> qt.Qobj:
    """Swap two qubits in a multi‑qubit operator."""
    if source == target:
        return op
    order = list(range(len(op.dims[0])))
    order[source], order[target] = order[target], order[source]
    return op.permute(order)


def _random_qubit_unitary(num_qubits: int) -> qt.Qobj:
    """Generate a random unitary on `num_qubits` qubits."""
    dim = 2 ** num_qubits
    matrix = sc.random.normal(size=(dim, dim)) + 1j * sc.random.normal(size=(dim, dim))
    unitary = sc.linalg.orth(matrix)
    qobj = qt.Qobj(unitary)
    dims = [2] * num_qubits
    qobj.dims = [dims.copy(), dims.copy()]
    return qobj


def _random_qubit_state(num_qubits: int) -> qt.Qobj:
    """Sample a random pure state on `num_qubits` qubits."""
    dim = 2 ** num_qubits
    amplitudes = sc.random.normal(size=(dim, 1)) + 1j * sc.random.normal(size=(dim, 1))
    amplitudes /= sc.linalg.norm(amplitudes)
    state = qt.Qobj(amplitudes)
    state.dims = [[2] * num_qubits, [1] * num_qubits]
    return state


def random_training_data(unitary: qt.Qobj, samples: int) -> List[Tuple[qt.Qobj, qt.Qobj]]:
    """Generate training pairs (state, unitary·state)."""
    dataset: List[Tuple[qt.Qobj, qt.Qobj]] = []
    num_qubits = len(unitary.dims[0])
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        dataset.append((state, unitary * state))
    return dataset


def random_network(qnn_arch: List[int], samples: int):
    """Build a random quantum feed‑forward network and training data."""
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


def _partial_trace_keep(state: qt.Qobj, keep: Sequence[int]) -> qt.Qobj:
    if len(keep)!= len(state.dims[0]):
        return state.ptrace(list(keep))
    return state


def _partial_trace_remove(state: qt.Qobj, remove: Sequence[int]) -> qt.Qobj:
    keep = list(range(len(state.dims[0])))
    for index in sorted(remove, reverse=True):
        keep.pop(index)
    return _partial_trace_keep(state, keep)


def _layer_channel(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[qt.Qobj]],
    layer: int,
    input_state: qt.Qobj,
) -> qt.Qobj:
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    state = qt.tensor(input_state, _tensored_zero(num_outputs))

    layer_unitary = unitaries[layer][0].copy()
    for gate in unitaries[layer][1:]:
        layer_unitary = gate * layer_unitary

    return _partial_trace_remove(layer_unitary * state * layer_unitary.dag(), range(num_inputs))


def feedforward(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[qt.Qobj]],
    samples: Iterable[Tuple[qt.Qobj, qt.Qobj]],
) -> List[List[qt.Qobj]]:
    """Run a forward pass through the quantum network for each sample."""
    stored_states: List[List[qt.Qobj]] = []
    for sample, _ in samples:
        layerwise = [sample]
        current_state = sample
        for layer in range(1, len(qnn_arch)):
            current_state = _layer_channel(qnn_arch, unitaries, layer, current_state)
            layerwise.append(current_state)
        stored_states.append(layerwise)
    return stored_states


def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
    """Absolute squared overlap between two pure states."""
    return abs((a.dag() * b)[0, 0]) ** 2


def fidelity_adjacency(
    states: Sequence[qt.Qobj],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Create a weighted graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


__all__ = [
    "QLSTM",
    "LSTMTagger",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
]
