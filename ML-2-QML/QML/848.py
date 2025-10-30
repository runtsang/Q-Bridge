import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import torchquantum as tq
import torchquantum.functional as tqf

class QLSTM(nn.Module):
    """Quantum LSTM with optional quantum feature extractor and quantum self‑attention."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0, attn_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Quantum gates for each LSTM gate
        self.forget_gate = self._build_gate(n_qubits)
        self.input_gate = self._build_gate(n_qubits)
        self.update_gate = self._build_gate(n_qubits)
        self.output_gate = self._build_gate(n_qubits)

        # Linear layers to map input+hidden to qubit dimension
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

        # Optional quantum feature extractor
        if n_qubits > 0:
            self.qfeat = self._build_qfeat(n_qubits)
        else:
            self.qfeat = None

        # Quantum self‑attention
        self.attn_gate = self._build_gate(n_qubits)

    def _build_gate(self, n_qubits: int) -> nn.Module:
        class QGate(tq.QuantumModule):
            def __init__(self, n_wires: int):
                super().__init__()
                self.n_wires = n_wires
                self.params = nn.Parameter(torch.randn(n_wires))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
                for i in range(self.n_wires):
                    tqf.rx(qdev, wires=[i], params=x[:, i % x.shape[1]])
                for i in range(self.n_wires):
                    tqf.rx(qdev, wires=[i], params=self.params[i])
                return tqf.measure_all(qdev, self.n_wires)

        return QGate(n_qubits)

    def _build_qfeat(self, n_qubits: int) -> nn.Module:
        class QFeatModule(tq.QuantumModule):
            def __init__(self, n_wires: int):
                super().__init__()
                self.n_wires = n_wires
                self.params = nn.Parameter(torch.randn(n_wires))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
                for i in range(self.n_wires):
                    tqf.rx(qdev, wires=[i], params=x[:, i % x.shape[1]])
                for i in range(self.n_wires):
                    tqf.rx(qdev, wires=[i], params=self.params[i])
                return tqf.measure_all(qdev, self.n_wires)

        return QFeatModule(n_qubits)

    def forward(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # inputs expected shape: (seq_len, batch, input_dim)
        if inputs.dim() == 3:
            seq_len, batch_size, _ = inputs.shape
            inputs = inputs.permute(1, 0, 2)  # (batch, seq_len, input_dim)
        else:
            batch_size, seq_len, _ = inputs.shape

        hx, cx = self._init_states(inputs, states)

        outputs = []
        for t in range(seq_len):
            x_t = inputs[:, t, :]
            combined = torch.cat([x_t, hx], dim=1)
            f = torch.sigmoid(self.forget_gate(self.linear_forget(combined)))
            i = torch.sigmoid(self.input_gate(self.linear_input(combined)))
            g = torch.tanh(self.update_gate(self.linear_update(combined)))
            o = torch.sigmoid(self.output_gate(self.linear_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)  # (batch, seq_len, hidden_dim)

        # Optional quantum feature extraction on hidden states
        if self.qfeat is not None:
            flat = outputs.reshape(-1, self.hidden_dim)
            qfeat_out = self.qfeat(flat)
            outputs = qfeat_out.reshape(batch_size, seq_len, -1)

        # Quantum self‑attention
        attn = torch.tanh(self.attn_gate(outputs))
        attn = torch.softmax(attn, dim=1)  # attention over sequence length
        context = torch.sum(attn * outputs, dim=1, keepdim=True)  # (batch, 1, hidden_dim)
        outputs = outputs + context  # broadcast addition

        return outputs, (hx, cx)

    def _init_states(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None):
        if states is not None:
            return states
        batch_size = inputs.size(0)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

class LSTMTagger(nn.Module):
    """Sequence tagging model that uses the quantum QLSTM."""
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int, tagset_size: int, n_qubits: int = 0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        # lstm_out shape: (batch, seq_len, hidden_dim)
        tag_logits = self.hidden2tag(lstm_out)  # shape: (batch, seq_len, tagset_size)
        tag_logits = tag_logits.permute(1, 0, 2)  # (seq_len, batch, tagset_size)
        return F.log_softmax(tag_logits, dim=2)

__all__ = ["QLSTM", "LSTMTagger"]
