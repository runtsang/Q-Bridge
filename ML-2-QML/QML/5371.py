"""
Quantum‑enhanced LSTM and associated utilities.
"""

from __future__ import annotations

from typing import Tuple, Iterable, Sequence, List, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import itertools

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector, Parameter
from qiskit.quantum_info import SparsePauliOp, Statevector
import qutip as qt

# --------------------------------------------------------------------------- #
# Quantum gate layer
# --------------------------------------------------------------------------- #
class QGate(nn.Module):
    """
    Variational gate that maps a hidden‑state vector to a set of
    Pauli‑Z expectation values.  The gate is parameterised by a
    small circuit of RX rotations followed by a ladder of CNOTs.
    """
    def __init__(self, n_qubits: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.params = ParameterVector(name="theta", length=n_qubits)

    def _build_circuit(self, param_values: torch.Tensor) -> QuantumCircuit:
        """Construct a circuit with the supplied parameters."""
        qc = QuantumCircuit(self.n_qubits)
        for i, val in enumerate(param_values.tolist()):
            qc.rx(val, i)
        # CNOT ladder to entangle qubits
        for i in range(self.n_qubits - 1):
            qc.cx(i, i + 1)
        return qc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Accepts a tensor of shape (batch, n_qubits) and returns
        a tensor of expectation values of Pauli‑Z on each qubit.
        """
        batch = x.shape[0]
        out = torch.zeros(batch, self.n_qubits, device=x.device, dtype=torch.float32)
        for b in range(batch):
            params = x[b]
            qc = self._build_circuit(params)
            sv = Statevector.from_instruction(qc)
            for q in range(self.n_qubits):
                pauli_z = SparsePauliOp.from_list([("I" * q + "Z" + "I" * (self.n_qubits - q - 1), 1)])
                exp = sv.expectation_value(pauli_z).real
                out[b, q] = exp
        return out

# --------------------------------------------------------------------------- #
# Quantum‑enhanced LSTM
# --------------------------------------------------------------------------- #
class QuantumQLSTM(nn.Module):
    """
    LSTM cell where each gate is realised by a `QGate`.  The hidden
    dimension must equal the number of qubits used by each gate.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        assert n_qubits == hidden_dim, "For this simplified design, n_qubits must equal hidden_dim."
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        self.forget_gate = QGate(n_qubits)
        self.input_gate = QGate(n_qubits)
        self.update_gate = QGate(n_qubits)
        self.output_gate = QGate(n_qubits)

        self.linear_forget = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.linear_input = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.linear_update = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.linear_output = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_gate(self.linear_forget(combined)))
            i = torch.sigmoid(self.input_gate(self.linear_input(combined)))
            g = torch.tanh(self.update_gate(self.linear_update(combined)))
            o = torch.sigmoid(self.output_gate(self.linear_output(combined)))
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

# --------------------------------------------------------------------------- #
# Tagging model wrapper
# --------------------------------------------------------------------------- #
class QLSTMTagger(nn.Module):
    """
    Wrapper that selects the classical `nn.LSTM` or the quantum `QuantumQLSTM`
    depending on the value of `n_qubits`.
    """
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
            self.lstm = QuantumQLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

# --------------------------------------------------------------------------- #
# Quantum classifier builder
# --------------------------------------------------------------------------- #
def build_classifier_circuit(
    num_qubits: int,
    depth: int,
) -> Tuple[QuantumCircuit, Iterable, Iterable, List[qt.Qobj]]:
    """
    Construct a layered ansatz with explicit encoding and variational parameters.
    Returns the circuit, the encoding parameters, the weight parameters,
    and a list of observables (Pauli‑Z on each qubit).
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    observables = [qt.tensor([qt.sigmaz() if i == j else qt.qeye(2) for i in range(num_qubits)]) for j in range(num_qubits)]
    return circuit, list(encoding), list(weights), observables

# --------------------------------------------------------------------------- #
# Quantum estimator QNN
# --------------------------------------------------------------------------- #
class EstimatorQNN(nn.Module):
    """
    Quantum estimator that wraps a single‑qubit circuit and a Qiskit
    `StatevectorEstimator`.  The circuit contains a parameterised
    Ry and Rx gate; the estimator returns the expectation of Pauli‑Y.
    """
    def __init__(self):
        super().__init__()
        self.params = [Parameter("input1"), Parameter("weight1")]
        self.circuit = QuantumCircuit(1)
        self.circuit.h(0)
        self.circuit.ry(self.params[0], 0)
        self.circuit.rx(self.params[1], 0)
        self.observable = SparsePauliOp.from_list([("Y", 1)])

    def forward(self, input_value: torch.Tensor) -> torch.Tensor:
        """
        Compute the expectation of Pauli‑Y given the current parameter
        values.  The `input_value` is mapped to the first circuit parameter.
        """
        bound_circuit = self.circuit.bind_parameters({self.params[0]: float(input_value.item()),
                                                     self.params[1]: 0.0})
        sv = Statevector.from_instruction(bound_circuit)
        exp = sv.expectation_value(self.observable).real
        return torch.tensor(exp, device=input_value.device, dtype=torch.float32)

# --------------------------------------------------------------------------- #
# Graph utilities (quantum)
# --------------------------------------------------------------------------- #
def _random_qubit_unitary(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    matrix = np.random.normal(size=(dim, dim)) + 1j * np.random.normal(size=(dim, dim))
    unitary = np.linalg.qr(matrix)[0]
    return qt.Qobj(unitary, dims=[[2]*num_qubits, [2]*num_qubits])

def _random_qubit_state(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    amplitude = np.random.normal(size=(dim, 1)) + 1j * np.random.normal(size=(dim, 1))
    amplitude /= np.linalg.norm(amplitude)
    return qt.Qobj(amplitude, dims=[[2]*num_qubits, [1]*num_qubits])

def random_training_data(unitary: qt.Qobj, samples: int) -> List[Tuple[qt.Qobj, qt.Qobj]]:
    dataset = []
    num_qubits = len(unitary.dims[0])
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        dataset.append((state, unitary * state))
    return dataset

def _tensored_id(num_qubits: int) -> qt.Qobj:
    return qt.qeye(2 ** num_qubits)

def _swap_registers(op: qt.Qobj, source: int, target: int) -> qt.Qobj:
    if source == target:
        return op
    perm = list(range(len(op.dims[0])))
    perm[source], perm[target] = perm[target], perm[source]
    return op.permute(perm)

def _partial_trace_keep(state: qt.Qobj, keep: Sequence[int]) -> qt.Qobj:
    if len(keep)!= len(state.dims[0]):
        return state.ptrace(list(keep))
    return state

def _partial_trace_remove(state: qt.Qobj, remove: Sequence[int]) -> qt.Qobj:
    keep = list(range(len(state.dims[0])))
    for index in sorted(remove, reverse=True):
        keep.pop(index)
    return _partial_trace_keep(state, keep)

def _layer_channel(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qt.Qobj]], layer: int, input_state: qt.Qobj) -> qt.Qobj:
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    state = qt.tensor(input_state, _tensored_id(num_outputs))
    layer_unitary = unitaries[layer][0].copy()
    for gate in unitaries[layer][1:]:
        layer_unitary = gate * layer_unitary
    return _partial_trace_remove(layer_unitary * state * layer_unitary.dag(), range(num_inputs))

def random_network(qnn_arch: List[int], samples: int):
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    unitaries: List[List[qt.Qobj]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        layer_ops: List[qt.Qobj] = []
        for _ in range(num_outputs):
            op = _random_qubit_unitary(num_inputs + 1)
            layer_ops.append(op)
        unitaries.append(layer_ops)

    return qnn_arch, unitaries, training_data, target_unitary

def feedforward(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[qt.Qobj]],
    samples: Iterable[Tuple[qt.Qobj, qt.Qobj]],
) -> List[List[qt.Qobj]]:
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
    """Return the squared overlap between two pure states."""
    return abs((a.dag() * b)[0, 0]) ** 2

def fidelity_adjacency(
    states: Sequence[qt.Qobj],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
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
    "QuantumQLSTM",
    "QLSTMTagger",
    "build_classifier_circuit",
    "EstimatorQNN",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
]
