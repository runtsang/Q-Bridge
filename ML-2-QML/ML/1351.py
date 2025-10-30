"""Hybrid classical‑quantum LSTM for sequence tagging – classical backbone with optional quantum gate modules."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import torchquantum
try:
    import torchquantum as tq
    import torchquantum.functional as tqf
except ImportError:
    tq = None
    tqf = None

class QLSTM__gen308(nn.Module):
    """Hybrid LSTM cell with optional quantum gates for each gate.

    Parameters
    ----------
    input_dim : int
        Size of input features.
    hidden_dim : int
        Size of hidden state.
    n_qubits : int, default 0
        Number of qubits used for quantum gates. If 0, the cell behaves as a classical LSTM.
    """

    class QuantumGate(nn.Module):
        """Variational circuit that returns a scalar gate value."""

        def __init__(self, n_qubits: int):
            super().__init__()
            self.n_qubits = n_qubits
            # Linear layer to map classical vector to qubit angles
            self.linear = nn.Linear(n_qubits, n_qubits)
            # Trainable parameters for the ansatz
            self.params = nn.Parameter(torch.randn(n_qubits))
            # Quantum device
            if tq is not None:
                self.qdevice = tq.QuantumDevice(n_wires=n_qubits, bsz=1)
            else:
                self.qdevice = None

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x shape: (batch, n_qubits)
            z = self.linear(x)  # (batch, n_qubits)
            if self.qdevice is None:
                # Fallback to a simple linear transform
                return torch.sigmoid(z.sum(dim=1, keepdim=True))
            # Encode data
            self.qdevice.reset()
            for i in range(self.n_qubits):
                self.qdevice.rx(z[:, i], wires=i)
            # Ansatz
            for i in range(self.n_qubits):
                self.qdevice.rx(self.params[i], wires=i)
            # Measure
            out = self.qdevice.measure_all()
            # Convert to gate value
            return torch.sigmoid(out.sum(dim=1, keepdim=True))

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Classical linear layers for gates
        self.forget_lin = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_lin = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_lin = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_lin = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # Quantum gates if requested
        if n_qubits > 0:
            self.forget_q = self.QuantumGate(n_qubits)
            self.input_q = self.QuantumGate(n_qubits)
            self.update_q = self.QuantumGate(n_qubits)
            self.output_q = self.QuantumGate(n_qubits)
        else:
            self.forget_q = None
            self.input_q = None
            self.update_q = None
            self.output_q = None

    def forward(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            # Classical gate outputs
            f = torch.sigmoid(self.forget_lin(combined))
            i = torch.sigmoid(self.input_lin(combined))
            g = torch.tanh(self.update_lin(combined))
            o = torch.sigmoid(self.output_lin(combined))
            if self.n_qubits > 0:
                # Quantum gate outputs
                f = torch.sigmoid(self.forget_q(self.forget_lin(combined)))
                i = torch.sigmoid(self.input_q(self.input_lin(combined)))
                g = torch.tanh(self.update_q(self.update_lin(combined)))
                o = torch.sigmoid(self.output_q(self.output_lin(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

class LSTMTagger(nn.Module):
    """Sequence tagging model that can switch between classical and quantum LSTM."""

    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int, tagset_size: int, n_qubits: int = 0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM__gen308(embedding_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTM__gen308", "LSTMTagger"]
