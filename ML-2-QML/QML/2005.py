import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from typing import Tuple, Optional

class QQuantumEncoder(tq.QuantumModule):
    """Variational encoder that maps a classical vector to a quantum feature vector."""
    def __init__(self, input_dim: int, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i % n_wires]} for i in range(input_dim)]
        )
        self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
        self.encoder(qdev, x)
        for wire, gate in enumerate(self.params):
            gate(qdev, wires=wire)
        return self.measure(qdev)

class QAttentionEncoder(tq.QuantumModule):
    """Quantum encoder used for attention score computation."""
    def __init__(self, hidden_dim: int, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i % n_wires]} for i in range(hidden_dim)]
        )
        self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
        self.encoder(qdev, x)
        for wire, gate in enumerate(self.params):
            gate(qdev, wires=wire)
        return self.measure(qdev)

class QLSTM(nn.Module):
    """Quantum‑enhanced LSTM cell with variational encoder and gated‑attention read‑out."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int, q_dim: int = 8):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.q_dim = q_dim

        # Quantum encoder for input
        self.quantum_encoder = QQuantumEncoder(input_dim, n_qubits)

        # Classical linear gates, operating on quantum‑encoded input concatenated with hidden state
        self.forget = nn.Linear(q_dim + hidden_dim, hidden_dim)
        self.input_gate = nn.Linear(q_dim + hidden_dim, hidden_dim)
        self.update = nn.Linear(q_dim + hidden_dim, hidden_dim)
        self.output_gate = nn.Linear(q_dim + hidden_dim, hidden_dim)

        # Attention read‑out
        self.attn_encoder = QAttentionEncoder(hidden_dim, n_qubits)
        self.attn_linear = nn.Linear(n_qubits, 1)

    def forward(self,
                inputs: torch.Tensor,
                states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            qfeat = self.quantum_encoder(x)  # (B, q_dim)
            combined = torch.cat([qfeat, hx], dim=1)
            f = torch.sigmoid(self.forget(combined))
            i = torch.sigmoid(self.input_gate(combined))
            g = torch.tanh(self.update(combined))
            o = torch.sigmoid(self.output_gate(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)  # seq_len, batch, hidden_dim
        return outputs, (hx, cx)

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

    def attention_readout(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Gated‑attention read‑out over the sequence of hidden states.
        hidden_states: (seq_len, batch, hidden_dim)
        Returns a context vector of shape (batch, hidden_dim).
        """
        seq_len, batch, _ = hidden_states.size()
        scores = []
        for t in range(seq_len):
            h_t = hidden_states[t]  # (batch, hidden_dim)
            qfeat = self.attn_encoder(h_t)  # (batch, n_qubits)
            score = self.attn_linear(qfeat).squeeze(-1)  # (batch,)
            scores.append(score)
        scores = torch.stack(scores, dim=1)  # (batch, seq_len)
        weights = F.softmax(scores, dim=1).unsqueeze(-1)  # (batch, seq_len, 1)
        hidden_states_t = hidden_states.transpose(0, 1)  # (batch, seq_len, hidden_dim)
        context = torch.sum(weights * hidden_states_t, dim=1)  # (batch, hidden_dim)
        return context

class LSTMTagger(nn.Module):
    """Sequence tagging model that can switch between classical and quantum LSTM."""
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 n_qubits: int = 0,
                 q_dim: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits, q_dim=q_dim)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

    def context_representation(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Return a global context vector via the quantum attention read‑out
        (only available when the underlying LSTM is quantum).
        """
        if not isinstance(self.lstm, QLSTM):
            raise RuntimeError("Context read‑out is only defined for the quantum LSTM.")
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        return self.lstm.attention_readout(lstm_out)

__all__ = ["QLSTM", "LSTMTagger"]
