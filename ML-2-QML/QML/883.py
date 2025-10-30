import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import pennylane.numpy as np
from typing import Tuple, Optional, List

class QuantumGate(nn.Module):
    """Quantum gate implemented as a variational circuit returning expectation values."""
    def __init__(self, n_wires: int, n_params: int):
        super().__init__()
        self.n_wires = n_wires
        self.n_params = n_params
        self.dev = qml.device("default.qubit", wires=n_wires, shots=0)

        @qml.qnode(self.dev, interface="torch")
        def circuit(x, params):
            # Encode classical input
            for i in range(n_wires):
                qml.RX(x[i], wires=i)
            # Variational parameters
            for i in range(n_params):
                qml.RY(params[i], wires=i % n_wires)
            # Entanglement
            for i in range(n_wires - 1):
                qml.CNOT(wires=[i, i + 1])
            # Measurement: expectation of PauliZ on each wire
            return [qml.expval(qml.PauliZ(i)) for i in range(n_wires)]

        self.circuit = circuit
        self.params = nn.Parameter(torch.randn(n_params))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, n_wires)
        return self.circuit(x, self.params)

class QLSTM(nn.Module):
    """Quantum LSTM cell where forget and input gates are quantum."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Linear projection to quantum space
        self.proj_to_q = nn.Linear(input_dim + hidden_dim, n_qubits, bias=False)

        # Quantum gates for forget and input
        self.forget_gate = QuantumGate(n_qubits, n_qubits)
        self.input_gate = QuantumGate(n_qubits, n_qubits)

        # Classical gates for update and output
        self.update_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(self, inputs: torch.Tensor, states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            qvec = self.proj_to_q(combined)
            f = torch.sigmoid(self.forget_gate(qvec))
            i = torch.sigmoid(self.input_gate(qvec))
            g = torch.tanh(self.update_gate(combined))
            o = torch.sigmoid(self.output_gate(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        return torch.cat(outputs, dim=0), (hx, cx)

    def _init_states(self, inputs: torch.Tensor, states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        if states is not None:
            return states
        batch_size = inputs.shape[1]
        device = inputs.device
        return torch.zeros(batch_size, self.hidden_dim, device=device), \
               torch.zeros(batch_size, self.hidden_dim, device=device)

class Tagger(nn.Module):
    """Sequence tagger that uses QLSTM for quantum sequence modelling."""
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int, tagset_size: int, n_qubits: int):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        return F.log_softmax(self.hidden2tag(lstm_out.view(len(sentence), -1)), dim=1)

__all__ = ["QLSTM", "Tagger"]
