import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class QLSTM(nn.Module):
    """Hybrid classical LSTM with optional quantum feature extractor and self‑attention."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0, attn_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Standard LSTM
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

        # Optional quantum feature extractor
        if n_qubits > 0:
            self.qfeat = self._build_qfeat(n_qubits)
        else:
            self.qfeat = None

        # Self‑attention layers
        self.attn_proj = nn.Linear(hidden_dim, hidden_dim)

    def _build_qfeat(self, n_qubits: int) -> nn.Module:
        import torchquantum as tq
        import torchquantum.functional as tqf

        class QFeatModule(tq.QuantumModule):
            def __init__(self, n_wires: int):
                super().__init__()
                self.n_wires = n_wires
                self.params = nn.Parameter(torch.randn(n_wires))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # x shape: (batch, hidden_dim)
                qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
                # Encode each feature into a rotation on a qubit
                for i in range(self.n_wires):
                    idx = i % x.shape[1]
                    tqf.rx(qdev, wires=[i], params=x[:, idx])
                # Apply trainable rotations
                for i in range(self.n_wires):
                    tqf.rx(qdev, wires=[i], params=self.params[i])
                # Measurement
                return tqf.measure_all(qdev, self.n_wires)

        return QFeatModule(n_wires=n_qubits)

    def forward(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # inputs expected shape: (seq_len, batch, input_dim)
        if inputs.dim() == 3:
            seq_len, batch_size, _ = inputs.shape
            inputs = inputs.permute(1, 0, 2)  # (batch, seq_len, input_dim)
        else:
            batch_size, seq_len, _ = inputs.shape

        # LSTM forward
        lstm_out, (hn, cn) = self.lstm(inputs, states)

        # Optional quantum feature extraction on hidden states
        if self.qfeat is not None:
            flat = lstm_out.reshape(-1, self.hidden_dim)
            qfeat_out = self.qfeat(flat)
            lstm_out = qfeat_out.reshape(batch_size, seq_len, -1)

        # Self‑attention
        attn = torch.tanh(self.attn_proj(lstm_out))
        attn = torch.softmax(attn, dim=1)  # attention over sequence length
        context = torch.sum(attn * lstm_out, dim=1, keepdim=True)  # (batch, 1, hidden_dim)
        lstm_out = lstm_out + context  # broadcast addition

        return lstm_out, (hn, cn)

class LSTMTagger(nn.Module):
    """Sequence tagging model that uses the hybrid QLSTM."""
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
