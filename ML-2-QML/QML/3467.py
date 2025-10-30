"""Quantum‑enhanced UnifiedGraphQLSTM.

This module implements a quantum‑aware version of the UnifiedGraphQLSTM
using torchquantum for the LSTM gates and qutip for a lightweight
graph quantum neural network.  The quantum GNN propagates qubit states
through random unitaries, while the QLSTMQuantum maps sequences of
amplitude vectors to hidden states via variational circuits.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable
from typing import List, Sequence, Tuple

import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import qutip as qt
import torchquantum as tq
import torchquantum.functional as tqf

Tensor = torch.Tensor

# --------------------------------------------------------------------------- #
# Quantum utilities (mirrors QLSTM seed)
# --------------------------------------------------------------------------- #
def _tensored_id(num_qubits: int) -> qt.Qobj:
    """Tensor product of identities."""
    identity = qt.qeye(2**num_qubits)
    dims = [2] * num_qubits
    identity.dims = [dims.copy(), dims.copy()]
    return identity

def _tensored_zero(num_qubits: int) -> qt.Qobj:
    """Tensor product of |0> projectors."""
    projector = qt.fock(2**num_qubits).proj()
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
    dim = 2**num_qubits
    matrix = np.random.normal(size=(dim, dim)) + 1j * np.random.normal(size=(dim, dim))
    unitary = np.linalg.qr(matrix)[0]
    qobj = qt.Qobj(unitary)
    dims = [2] * num_qubits
    qobj.dims = [dims.copy(), dims.copy()]
    return qobj

def _random_qubit_state(num_qubits: int) -> qt.Qobj:
    dim = 2**num_qubits
    amplitudes = np.random.normal(size=(dim, 1)) + 1j * np.random.normal(size=(dim, 1))
    amplitudes /= np.linalg.norm(amplitudes)
    state = qt.Qobj(amplitudes)
    state.dims = [[2] * num_qubits, [1] * num_qubits]
    return state

def random_training_data(
    unitary: qt.Qobj, samples: int
) -> List[Tuple[qt.Qobj, qt.Qobj]]:
    """Generate training pairs (state, unitary*state)."""
    dataset: List[Tuple[qt.Qobj, qt.Qobj]] = []
    num_qubits = len(unitary.dims[0])
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        dataset.append((state, unitary * state))
    return dataset

def random_network(arch: List[int], samples: int):
    """Produce a toy quantum network of random unitaries and a training set."""
    target_unitary = _random_qubit_unitary(arch[-1])
    training = random_training_data(target_unitary, samples)

    unitaries: List[List[qt.Qobj]] = [[]]
    for layer in range(1, len(arch)):
        num_inputs = arch[layer - 1]
        num_outputs = arch[layer]
        layer_ops: List[qt.Qobj] = []
        for output in range(num_outputs):
            op = _random_qubit_unitary(num_inputs + 1)
            if num_outputs > 1:
                op = qt.tensor(_random_qubit_unitary(num_inputs + 1), _tensored_id(num_outputs - 1))
                op = _swap_registers(op, num_inputs, num_inputs + output)
            layer_ops.append(op)
        unitaries.append(layer_ops)

    return arch, unitaries, training, target_unitary

def _partial_trace_keep(state: qt.Qobj, keep: Sequence[int]) -> qt.Qobj:
    if len(keep)!= len(state.dims[0]):
        return state.ptrace(list(keep))
    return state

def _partial_trace_remove(state: qt.Qobj, remove: Sequence[int]) -> qt.Qobj:
    keep = list(range(len(state.dims[0])))
    for idx in sorted(remove, reverse=True):
        keep.pop(idx)
    return _partial_trace_keep(state, keep)

def _layer_channel(
    arch: Sequence[int],
    unitaries: Sequence[Sequence[qt.Qobj]],
    layer: int,
    input_state: qt.Qobj,
) -> qt.Qobj:
    num_inputs = arch[layer - 1]
    num_outputs = arch[layer]
    state = qt.tensor(input_state, _tensored_zero(num_outputs))

    layer_unitary = unitaries[layer][0].copy()
    for gate in unitaries[layer][1:]:
        layer_unitary = gate * layer_unitary

    return _partial_trace_remove(layer_unitary * state * layer_unitary.dag(), range(num_inputs))

def feedforward(
    arch: Sequence[int], unitaries: Sequence[Sequence[qt.Qobj]], samples: Iterable[Tuple[qt.Qobj, qt.Qobj]]
) -> List[List[qt.Qobj]]:
    """Forward propagation through the quantum network."""
    stored_states: List[List[qt.Qobj]] = []
    for state, _ in samples:
        layerwise = [state]
        current = state
        for layer in range(1, len(arch)):
            current = _layer_channel(arch, unitaries, layer, current)
            layerwise.append(current)
        stored_states.append(layerwise)
    return stored_states

def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
    """Squared overlap between pure quantum states."""
    return abs((a.dag() * b)[0, 0]) ** 2

def fidelity_adjacency(
    states: Sequence[qt.Qobj],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Build a weighted graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(a, b)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

# --------------------------------------------------------------------------- #
# Quantum LSTM cell (mirrors QLSTM seed)
# --------------------------------------------------------------------------- #
class QLSTMQuantum(nn.Module):
    """LSTM cell where gates are realised by small quantum circuits."""

    class _QLayer(tq.QuantumModule):
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
            self.params = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: Tensor) -> Tensor:
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

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        self.forget = self._QLayer(n_qubits)
        self.input = self._QLayer(n_qubits)
        self.update = self._QLayer(n_qubits)
        self.output = self._QLayer(n_qubits)

        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(
        self,
        inputs: Tensor,
        states: Tuple[Tensor, Tensor] | None = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs: List[Tensor] = []
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
        inputs: Tensor,
        states: Tuple[Tensor, Tensor] | None,
    ) -> Tuple[Tensor, Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

# --------------------------------------------------------------------------- #
# Quantum GNN
# --------------------------------------------------------------------------- #
class QuantumGNN(nn.Module):
    """Lightweight quantum graph neural network using random unitaries."""
    def __init__(self, arch: Sequence[int], device: torch.device | None = None):
        super().__init__()
        self.arch = arch
        self.device = device
        _, self.unitaries, _, self.target = random_network(list(arch), samples=0)

    def forward(self, graph: nx.Graph) -> List[qt.Qobj]:
        """Return a list of quantum states, one per node."""
        num_nodes = graph.number_of_nodes()
        # Random state per node; graph structure is ignored in this placeholder.
        states = [_random_qubit_state(self.arch[-1]) for _ in range(num_nodes)]
        return states

# --------------------------------------------------------------------------- #
# Unified quantum‑classical hybrid
# --------------------------------------------------------------------------- #
class UnifiedGraphQLSTM(nn.Module):
    """Quantum‑aware hybrid model using QuantumGNN and QLSTMQuantum."""
    def __init__(
        self,
        gnn_arch: Sequence[int],
        lstm_input_dim: int,
        lstm_hidden_dim: int,
        lstm_output_dim: int,
        n_qubits: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        if gnn_arch[0]!= lstm_input_dim:
            raise ValueError("gnn_arch[0] must equal lstm_input_dim")
        self.gnn = QuantumGNN(gnn_arch, device=device)
        self.lstm = QLSTMQuantum(lstm_input_dim, lstm_hidden_dim, n_qubits)
        self.final = nn.Linear(lstm_hidden_dim, lstm_output_dim)

    def _qobj_to_tensor(self, qobj: qt.Qobj) -> Tensor:
        arr = qobj.data.to_numpy()
        real = torch.tensor(np.real(arr), dtype=torch.float32)
        imag = torch.tensor(np.imag(arr), dtype=torch.float32)
        return torch.cat([real, imag], dim=0)

    def forward(self, graph: nx.Graph) -> Tensor:
        """Run a graph through the quantum GNN, convert states to amplitudes,
        feed into the quantum LSTM, and produce logits."""
        states = self.gnn(graph)
        features = torch.stack([self._qobj_to_tensor(s) for s in states], dim=0)
        seq = features.unsqueeze(1)  # (N,1,D)
        lstm_out, _ = self.lstm(seq)
        out = self.final(lstm_out.squeeze(1))
        return out

__all__ = [
    "UnifiedGraphQLSTM",
    "QLSTMQuantum",
    "QuantumGNN",
    "state_fidelity",
    "fidelity_adjacency",
]
