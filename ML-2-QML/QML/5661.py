"""
Hybrid classifier and LSTM module with a quantum implementation.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

__all__ = ["HybridClassifierQLSTM", "build_quantum_classifier",
           "build_quantum_lstm", "QuantumLSTMTagger"]


def build_quantum_classifier(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """
    Construct a data‑uploading quantum classifier circuit.
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

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return circuit, list(encoding), list(weights), observables


class QuantumLSTMTagger(nn.Module):
    """
    Sequence tagging model that uses quantum gates for the LSTM gates.
    """
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

        self.forget = self.QLayer(n_qubits)
        self.input = self.QLayer(n_qubits)
        self.update = self.QLayer(n_qubits)
        self.output = self.QLayer(n_qubits)

        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] = None):
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


def build_quantum_lstm(embedding_dim: int,
                       hidden_dim: int,
                       vocab_size: int,
                       tagset_size: int,
                       n_qubits: int) -> nn.Module:
    """
    Factory that returns a QuantumLSTMTagger instance.
    """
    class QuantumLSTMWrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.hidden_dim = hidden_dim
            self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
            self.lstm = QuantumLSTMTagger(embedding_dim, hidden_dim, n_qubits)
            self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

        def forward(self, sentence: torch.Tensor) -> torch.Tensor:
            embeds = self.word_embeddings(sentence)
            lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
            tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
            return F.log_softmax(tag_logits, dim=1)
    return QuantumLSTMWrapper()


class HybridClassifierQLSTM:
    """
    Hybrid implementation of the classifier and LSTM that can switch between
    classical and quantum components via the `use_quantum` flag.
    """
    def __init__(self,
                 num_features: int,
                 classifier_depth: int,
                 num_qubits: int,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 use_quantum: bool = True,
                 n_qubits_lstm: int = 4) -> None:
        self.use_quantum = use_quantum

        if use_quantum:
            self.classifier, self.encoding, self.weight_sizes, self.observables = build_quantum_classifier(
                num_qubits, classifier_depth
            )
            self.lstm = build_quantum_lstm(embedding_dim, hidden_dim, vocab_size, tagset_size, n_qubits_lstm)
        else:
            # Import the classical fallback from the ml module
            from. import build_classical_classifier, build_classical_lstm
            self.classifier, self.encoding, self.weight_sizes, self.observables = build_classical_classifier(
                num_features, classifier_depth
            )
            self.lstm = build_classical_lstm(embedding_dim, hidden_dim, vocab_size, tagset_size)

    def forward_classifier(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the classifier.
        """
        return self.classifier(x)

    def forward_lstm(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the LSTM tagger.
        """
        return self.lstm(sentence)
