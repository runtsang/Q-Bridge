"""Hybrid quantum LSTM with QCNN feature extractor and quantum gates.

This module defines a QLSTMGen195 class that uses torchquantum for the LSTM gates
and a qiskit‑based QCNN as a feature extractor.  The forward pass returns the
hidden‑state sequence, the final hidden/cell states and a fidelity‑based adjacency
graph of the hidden states.  The API mirrors the original QLSTM and LSTMTagger
classes for compatibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np
import torchquantum as tq
import torchquantum.functional as tqf
from typing import Tuple, List, Optional

# --------------------------------------------------------------------------- #
# State fidelity & adjacency (borrow from GraphQNN)
# --------------------------------------------------------------------------- #
def _state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)

def fidelity_adjacency(states: List[torch.Tensor], threshold: float,
                       *, secondary: Optional[float] = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(len(states)))
    for (i, si), (j, sj) in zip(enumerate(states), enumerate(states[1:])):
        fid = _state_fidelity(si, sj)
        if fid >= threshold:
            G.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            G.add_edge(i, j, weight=secondary_weight)
    return G

# --------------------------------------------------------------------------- #
# QCNN builder (quantum convolutional network)
# --------------------------------------------------------------------------- #
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals

def QCNN() -> EstimatorQNN:
    algorithm_globals.random_seed = 12345
    estimator = Estimator()

    def conv_circuit(params):
        target = QuantumCircuit(2)
        target.rz(-np.pi / 2, 1)
        target.cx(1, 0)
        target.rz(params[0], 0)
        target.ry(params[1], 1)
        target.cx(0, 1)
        target.ry(params[2], 1)
        target.cx(1, 0)
        target.rz(np.pi / 2, 0)
        return target

    def conv_layer(num_qubits, param_prefix):
        qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
        qubits = list(range(num_qubits))
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits * 3)
        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            qc = qc.compose(conv_circuit(params[param_index : param_index + 3]), [q1, q2])
            qc.barrier()
            param_index += 3
        for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
            qc = qc.compose(conv_circuit(params[param_index : param_index + 3]), [q1, q2])
            qc.barrier()
            param_index += 3
        return qc

    def pool_circuit(params):
        target = QuantumCircuit(2)
        target.rz(-np.pi / 2, 1)
        target.cx(1, 0)
        target.rz(params[0], 0)
        target.ry(params[1], 1)
        target.cx(0, 1)
        target.ry(params[2], 1)
        return target

    def pool_layer(sources, sinks, param_prefix):
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits, name="Pooling Layer")
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
        for source, sink in zip(sources, sinks):
            qc = qc.compose(pool_circuit(params[param_index : param_index + 3]), [source, sink])
            qc.barrier()
            param_index += 3
        return qc

    feature_map = ZFeatureMap(8)
    ansatz = QuantumCircuit(8, name="Ansatz")
    ansatz.compose(conv_layer(8, "c1"), list(range(8)), inplace=True)
    ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), list(range(8)), inplace=True)
    ansatz.compose(conv_layer(4, "c2"), list(range(4, 8)), inplace=True)
    ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), list(range(4, 8)), inplace=True)
    ansatz.compose(conv_layer(2, "c3"), list(range(6, 8)), inplace=True)
    ansatz.compose(pool_layer([0], [1], "p3"), list(range(6, 8)), inplace=True)

    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, range(8), inplace=True)
    circuit.compose(ansatz, range(8), inplace=True)

    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )
    return qnn

# --------------------------------------------------------------------------- #
# Quantum LSTM gate layer
# --------------------------------------------------------------------------- #
class QLayer(tq.QuantumModule):
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

# --------------------------------------------------------------------------- #
# Hybrid quantum LSTM with QCNN feature extractor
# --------------------------------------------------------------------------- #
class QLSTMGen195(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 n_qubits: int = 0,
                 use_qcnn: bool = False) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.use_qcnn = use_qcnn

        self.forget = self.QLayer(n_qubits)
        self.input_gate = self.QLayer(n_qubits)
        self.update = self.QLayer(n_qubits)
        self.output = self.QLayer(n_qubits)

        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

        self.feature_extractor: nn.Module = QCNN() if use_qcnn else nn.Identity()

    def forward(self,
                inputs: torch.Tensor,
                states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
               ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], nx.Graph]:
        hx, cx = self._init_states(inputs, states)
        outputs: List[torch.Tensor] = []
        hidden_states: List[torch.Tensor] = []

        for x in inputs.unbind(dim=0):
            if self.use_qcnn:
                x = self.feature_extractor(x.unsqueeze(0))
                x = x.squeeze(0)
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(self.linear_forget(combined)))
            i = torch.sigmoid(self.input_gate(self.linear_input(combined)))
            g = torch.tanh(self.update(self.linear_update(combined)))
            o = torch.sigmoid(self.output(self.linear_output(combined)))

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)

            outputs.append(hx.unsqueeze(0))
            hidden_states.append(hx)

        adjacency = fidelity_adjacency(hidden_states, threshold=0.9)

        return torch.cat(outputs, dim=0), (hx, cx), adjacency

    def _init_states(self,
                     inputs: torch.Tensor,
                     states: Optional[Tuple[torch.Tensor, torch.Tensor]]
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

# --------------------------------------------------------------------------- #
# Tagger wrapper (compatible with original API)
# --------------------------------------------------------------------------- #
class LSTMTagger(nn.Module):
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 n_qubits: int = 0,
                 use_qcnn: bool = False) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTMGen195(embedding_dim, hidden_dim,
                                n_qubits=n_qubits,
                                use_qcnn=use_qcnn)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        embeds = self.word_embeddings(sentence)
        lstm_out, _, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
QLSTM = QLSTMGen195
__all__ = ["QLSTMGen195", "QLSTM", "LSTMTagger"]
