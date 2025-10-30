"""Quantum‑enhanced LSTM with a variational classifier head."""
from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from typing import Tuple, Iterable, List


def build_classifier_circuit_qiskit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """
    Construct a layered ansatz with explicit encoding and variational parameters.
    Mirrors the API of the classical helper but returns a Qiskit circuit.
    """
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

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return circuit, list(encoding), list(weights), observables


class QuantumQLSTMCell(tq.QuantumModule):
    """
    Quantum‑LSTM cell where each gate is realised by a small variational
    circuit.  The cell still produces classical tensors that can be used
    downstream in a PyTorch pipeline.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int, depth: int = 1) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.depth = depth

        # Linear layers that project classical concatenated state to qubit space
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

        # Quantum gates for each gate
        self.forget_gate = self._make_gate()
        self.input_gate = self._make_gate()
        self.update_gate = self._make_gate()
        self.output_gate = self._make_gate()

    def _make_gate(self) -> tq.QuantumModule:
        class Gate(tq.QuantumModule):
            def __init__(self, n_wires: int) -> None:
                super().__init__()
                self.n_wires = n_wires
                self.encoder = tq.GeneralEncoder(
                    [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
                )
                self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
                self.measure = tq.MeasureAll(tq.PauliZ)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
                self.encoder(qdev, x)
                for gate in self.params:
                    gate(qdev)
                # Entangling layer
                for wire in range(self.n_wires - 1):
                    tqf.cnot(qdev, wires=[wire, wire + 1])
                return self.measure(qdev)
        return Gate(self.n_qubits)

    def forward(self, x: torch.Tensor, hx: torch.Tensor, cx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([x, hx], dim=1)
        f = torch.sigmoid(self.forget_gate(self.linear_forget(combined)))
        i = torch.sigmoid(self.input_gate(self.linear_input(combined)))
        g = torch.tanh(self.update_gate(self.linear_update(combined)))
        o = torch.sigmoid(self.output_gate(self.linear_output(combined)))
        cx = f * cx + i * g
        hx = o * torch.tanh(cx)
        return hx, cx


class QuantumHybridQLSTMTagger(nn.Module):
    """
    Sequence tagging model that uses a quantum‑LSTM cell and a variational
    classifier head built with Qiskit.  The classifier can be queried
    as a standalone quantum circuit for interpretability or benchmarking.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        lstm_depth: int = 1,
        classifier_depth: int = 1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        if n_qubits > 0:
            self.lstm_cell = QuantumQLSTMCell(embedding_dim, hidden_dim, n_qubits, depth=lstm_depth)
        else:
            self.lstm_cell = nn.LSTMCell(embedding_dim, hidden_dim)

        # Build a quantum classifier head
        self.classifier_circuit, self.classifier_params, self.classifier_weights, self.classifier_observables = \
            build_classifier_circuit_qiskit(n_qubits, classifier_depth)

        # A classical linear layer to map hidden states to tag logits
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        ``sentence`` is a 1‑D tensor of token indices.
        Returns log‑softmaxed tag logits for each token.
        """
        embeds = self.word_embeddings(sentence)
        hx = torch.zeros(sentence.size(0), self.hidden_dim, device=embeds.device)
        cx = torch.zeros(sentence.size(0), self.hidden_dim, device=embeds.device)

        outputs: List[torch.Tensor] = []
        for x in embeds:
            hx, cx = self.lstm_cell(x, hx, cx)
            outputs.append(hx.unsqueeze(0))

        lstm_out = torch.cat(outputs, dim=0)
        logits = self.hidden2tag(lstm_out.view(-1, self.hidden_dim))
        logits = logits.view(len(sentence), -1, self.hidden2tag.out_features)
        return torch.log_softmax(logits, dim=2)

    def get_classifier_circuit(self) -> QuantumCircuit:
        """
        Return the underlying Qiskit circuit representing the classifier head.
        Useful for simulation or hardware execution.
        """
        return self.classifier_circuit

__all__ = ["QuantumHybridQLSTMTagger", "QuantumQLSTMCell", "build_classifier_circuit_qiskit"]
