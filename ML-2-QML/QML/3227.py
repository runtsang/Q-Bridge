import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import qutip as qt
import scipy as sc
import networkx as nx
import itertools
from typing import List, Tuple, Iterable, Sequence

Tensor = torch.Tensor
QState = qt.Qobj

class QLSTM(nn.Module):
    """
    Quantum‑enhanced LSTM where each gate is a small variational circuit.
    """
    class QGate(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            # Encode each input feature into an RX gate
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.rx_params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: Tensor) -> Tensor:
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
            self.encoder(qdev, x)
            for gate in self.rx_params:
                gate(qdev)
            # Simple linear‑chain entanglement
            for i in range(self.n_wires - 1):
                tqf.cnot(qdev, wires=[i, i + 1])
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Quantum gate modules
        self.forget_gate = self.QGate(n_qubits)
        self.input_gate = self.QGate(n_qubits)
        self.update_gate = self.QGate(n_qubits)
        self.output_gate = self.QGate(n_qubits)

        # Classical linear projections to qubit space
        self.forget_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.input_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.update_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.output_lin = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(self, inputs: Tensor, states: Tuple[Tensor, Tensor] | None = None) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_gate(self.forget_lin(combined)))
            i = torch.sigmoid(self.input_gate(self.input_lin(combined)))
            g = torch.tanh(self.update_gate(self.update_lin(combined)))
            o = torch.sigmoid(self.output_gate(self.output_lin(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        return torch.cat(outputs, dim=0), (hx, cx)

    def _init_states(self, inputs: Tensor, states: Tuple[Tensor, Tensor] | None):
        if states is not None:
            return states
        batch = inputs.size(1)
        device = inputs.device
        return torch.zeros(batch, self.hidden_dim, device=device), torch.zeros(batch, self.hidden_dim, device=device)

def random_network(qnn_arch: Sequence[int], samples: int):
    """
    Construct a random quantum neural network architecture and a target unitary.
    """
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = _random_training_data(target_unitary, samples)
    unitaries = [_random_layer_unitary(qnn_arch[i], qnn_arch[i + 1]) for i in range(len(qnn_arch) - 1)]
    return list(qnn_arch), unitaries, training_data, target_unitary

def _random_qubit_unitary(num_qubits: int) -> QState:
    dim = 2 ** num_qubits
    mat = sc.linalg.orth(sc.random.normal(size=(dim, dim)) + 1j * sc.random.normal(size=(dim, dim)))
    return qt.Qobj(mat, dims=[[2] * num_qubits, [2] * num_qubits])

def _random_qubit_state(num_qubits: int) -> QState:
    dim = 2 ** num_qubits
    vec = sc.random.normal(size=(dim, 1)) + 1j * sc.random.normal(size=(dim, 1))
    vec /= sc.linalg.norm(vec)
    return qt.Qobj(vec, dims=[[2] * num_qubits, [1] * num_qubits])

def _random_training_data(unitary: QState, samples: int) -> List[Tuple[QState, QState]]:
    data = []
    for _ in range(samples):
        state = _random_qubit_state(unitary.dims[0][0])
        data.append((state, unitary * state))
    return data

def _random_layer_unitary(n_in: int, n_out: int) -> List[QState]:
    """Return a list of unitaries modelling a single layer."""
    return [_random_qubit_unitary(n_in + 1) for _ in range(n_out)]

def feedforward(qnn_arch: Sequence[int], unitaries: Sequence[List[QState]], samples: Iterable[Tuple[QState, QState]]):
    """Propagate a batch of input states through the quantum network."""
    outputs = []
    for inp, _ in samples:
        state = inp
        layer_states = [state]
        for layer, ops in enumerate(unitaries, start=1):
            state = _apply_layer(state, ops)
            layer_states.append(state)
        outputs.append(layer_states)
    return outputs

def _apply_layer(state: QState, ops: List[QState]) -> QState:
    """Apply a layer of gates to the input state and partial‑trace."""
    zero = _tensored_zero(len(ops[0].dims[0]) - 1)
    full = qt.tensor(state, zero)
    U = ops[0]
    for op in ops[1:]:
        U = op * U
    new_state = U * full * U.dag()
    return _partial_trace_remove(new_state, list(range(len(state.dims[0]))))

def _tensored_id(num_qubits: int) -> QState:
    return qt.qeye(2 ** num_qubits)

def _tensored_zero(num_qubits: int) -> QState:
    return qt.fock(2 ** num_qubits).proj()

def _partial_trace_remove(state: QState, remove: List[int]) -> QState:
    keep = list(range(len(state.dims[0])))
    for idx in sorted(remove, reverse=True):
        keep.pop(idx)
    return state.ptrace(keep)

def state_fidelity(a: QState, b: QState) -> float:
    """Squared overlap between two pure states."""
    return abs((a.dag() * b)[0, 0]) ** 2

def fidelity_adjacency(states: Sequence[QState], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
    """Build a weighted graph from fidelities."""
    G = nx.Graph()
    G.add_nodes_from(range(len(states)))
    for i, a in enumerate(states):
        for j, b in enumerate(states[i + 1:], start=i + 1):
            fid = state_fidelity(a, b)
            if fid >= threshold:
                G.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                G.add_edge(i, j, weight=secondary_weight)
    return G

__all__ = [
    "QLSTM",
    "random_network",
    "feedforward",
    "fidelity_adjacency",
    "state_fidelity",
]
