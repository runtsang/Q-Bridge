import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from typing import Tuple, Optional

class QLayer(tq.QuantumModule):
    """Parameter‑sharded variational circuit with tunable depth."""
    def __init__(self, n_wires: int, depth: int = 2) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.depth = depth
        # Encoder: simple RX rotations per input feature
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "rx", "wires": [i]}
                for i in range(n_wires)
            ]
        )
        # Trainable parameters per depth layer
        self.params = nn.ModuleList([
            nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
            for _ in range(depth)
        ])
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
        self.encoder(qdev, x)
        for layer in self.params:
            for wire, gate in enumerate(layer):
                gate(qdev, wires=wire)
            # CNOT ladder
            for wire in range(self.n_wires):
                tgt = (wire + 1) % self.n_wires
                tqf.cnot(qdev, wires=[wire, tgt])
        return self.measure(qdev)

class QLSTM(nn.Module):
    """Base quantum LSTM where gates are realised by QLayer."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int, depth: int = 2) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.depth = depth

        self.forget = QLayer(n_qubits, depth)
        self.input = QLayer(n_qubits, depth)
        self.update = QLayer(n_qubits, depth)
        self.output = QLayer(n_qubits, depth)

        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None = None):
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

    def _init_states(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

class QLSTMGen(QLSTM):
    """Quantum LSTM with multi‑head attention over the hidden state."""
    def __init__(self, *args, **kwargs):
        self.attn_heads = kwargs.pop('attn_heads', 2)
        super().__init__(*args, **kwargs)
        self.attn_proj = nn.Linear(self.hidden_dim, self.hidden_dim * self.attn_heads, bias=False)
        self.attn_out = nn.Linear(self.hidden_dim * self.attn_heads, self.hidden_dim, bias=False)

    def forward(self, inputs: torch.Tensor, states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        outputs, (h, c) = super().forward(inputs, states)
        seq_len, batch_size, hidden_dim = outputs.shape
        attn_scores = self.attn_proj(outputs)
        attn_scores = attn_scores.view(seq_len, batch_size, self.attn_heads, hidden_dim)
        attn_weights = torch.softmax(attn_scores, dim=0)
        context = torch.sum(attn_weights * outputs.unsqueeze(2), dim=0)
        context = context.transpose(0,1).reshape(batch_size, self.attn_heads*hidden_dim)
        context = self.attn_out(context)
        context = context.unsqueeze(0).expand(seq_len, -1, -1)
        outputs = outputs + context
        return outputs, (h, c)

class LSTMTagger(nn.Module):
    """Sequence tagging model that can switch between classical and quantum LSTM."""
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int, tagset_size: int, n_qubits: int = 0, depth: int = 2, **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTMGen(embedding_dim, hidden_dim, n_qubits=n_qubits, depth=depth, **kwargs)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTM", "QLSTMGen", "LSTMTagger"]
