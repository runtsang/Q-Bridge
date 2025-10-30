"""Quantum LSTM tagger with quantum EstimatorQNN for output logits."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator


class QLSTMGen128(nn.Module):
    """Hybrid LSTM that uses quantum gates and a quantum EstimatorQNN for output logits."""
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int, tagset_size: int,
                 n_qubits: int = 0, generation_length: int = 128) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.generation_length = generation_length
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = self._QuantumLSTM(embedding_dim, hidden_dim, n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.estimator_qnn = self._build_estimator_qnn(hidden_dim)

    class _QuantumLSTM(nn.Module):
        """Quantum-enhanced LSTM cell."""
        def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
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

        class QLayer(tq.QuantumModule):
            def __init__(self, n_wires: int):
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
                for wire, gate in enumerate(self.params):
                    gate(qdev, wires=wire)
                for wire in range(self.n_wires - 1):
                    tqf.cnot(qdev, wires=[wire, wire + 1])
                tqf.cnot(qdev, wires=[self.n_wires - 1, 0])
                return self.measure(qdev)

        def forward(self, inputs: torch.Tensor, states: tuple | None = None):
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
            return torch.cat(outputs, dim=0), (hx, cx)

        def _init_states(self, inputs: torch.Tensor, states: tuple | None):
            if states is not None:
                return states
            batch_size = inputs.size(1)
            device = inputs.device
            return torch.zeros(batch_size, self.hidden_dim, device=device), torch.zeros(batch_size, self.hidden_dim, device=device)

    def _build_estimator_qnn(self, hidden_dim: int) -> EstimatorQNN:
        """Construct a quantum EstimatorQNN that maps hidden states to expectation values."""
        params = [Parameter(f"w{i}") for i in range(hidden_dim)]
        qc = QuantumCircuit(hidden_dim)
        for i, p in enumerate(params):
            qc.ry(p, i)
        # Measure all qubits to obtain a vector of expectation values
        observables = [SparsePauliOp.from_list([("Z" * hidden_dim, 1)])]
        estimator = StatevectorEstimator()
        return EstimatorQNN(circuit=qc,
                            observables=observables,
                            input_params=params,
                            weight_params=[],
                            estimator=estimator)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        # Flatten hidden states for estimator
        hidden_flat = lstm_out.view(-1, self.hidden_dim)
        quantum_logits = self.estimator_qnn(hidden_flat)
        logits = self.hidden2tag(quantum_logits)
        logits = logits.view(len(sentence), -1)
        return F.log_softmax(logits, dim=1)

    def generate(self, start_token: int, vocab_size: int) -> torch.Tensor:
        """Greedy generation using quantum estimator for logits."""
        generated = [start_token]
        hidden = None
        for _ in range(self.generation_length):
            token_tensor = torch.tensor([generated[-1]], device=self.word_embeddings.weight.device)
            embed = self.word_embeddings(token_tensor).unsqueeze(0).unsqueeze(0)
            lstm_out, hidden = self.lstm(embed, hidden)
            hidden_flat = lstm_out.squeeze(0)
            quantum_logits = self.estimator_qnn(hidden_flat)
            logits = self.hidden2tag(quantum_logits)
            next_token = torch.argmax(logits, dim=-1).item()
            generated.append(next_token)
        return torch.tensor(generated, dtype=torch.long)

__all__ = ["QLSTMGen128"]
