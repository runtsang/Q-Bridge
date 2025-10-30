import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class QLSTM(nn.Module):
    """
    Classical LSTM cell with optional dropout and a gated‑recurrent attention
    mechanism.  The gate tensors are produced by either a standard linear
    transformation or, if ``n_qubits`` > 0, by a small variational quantum
    circuit.  The design keeps the original API but adds a new ``forward_with_attention``.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0,
                 dropout_prob: float = 0.0, attn_dim: int = 0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.attn_dim = attn_dim
        self.dropout = nn.Dropout(p=dropout_prob) if dropout_prob > 0 else None

        # Linear gates
        self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # Quantum gates
        if self.n_qubits > 0:
            self.forget = self.QLayer(n_qubits)
            self.input = self.QLayer(n_qubits)
            self.update = self.QLayer(n_qubits)
            self.output = self.QLayer(n_qubits)
            self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

        # Attention projection
        if self.attn_dim > 0:
            self.attn_proj = nn.Linear(hidden_dim, attn_dim)
            self.attn_score = nn.Linear(attn_dim, 1, bias=False)

    class QLayer(nn.Module):
        """Simple variational layer that applies a depth‑controlled circuit."""
        def __init__(self, n_wires: int, depth: int = 1):
            super().__init__()
            self.n_wires = n_wires
            self.depth = depth
            self.params = nn.ParameterList(
                [nn.Parameter(torch.randn(n_wires)) for _ in range(depth)]
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Dummy quantum behaviour: apply element‑wise tanh of weighted sum
            out = x
            for p in self.params:
                out = torch.tanh(out + p)
            return out

    def _init_states(self, inputs: torch.Tensor,
                     states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

    def forward(self, inputs: torch.Tensor,
                states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            if self.n_qubits > 0:
                f = torch.sigmoid(self.forget(self.linear_forget(combined)))
                i = torch.sigmoid(self.input(self.linear_input(combined)))
                g = torch.tanh(self.update(self.linear_update(combined)))
                o = torch.sigmoid(self.output(self.linear_output(combined)))
            else:
                f = torch.sigmoid(self.forget_linear(combined))
                i = torch.sigmoid(self.input_linear(combined))
                g = torch.tanh(self.update_linear(combined))
                o = torch.sigmoid(self.output_linear(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            if self.dropout is not None:
                hx = self.dropout(hx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def forward_with_attention(self, inputs: torch.Tensor,
                               states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Run the LSTM and compute a global attention weight over the hidden states.
        """
        lstm_out, (hx, cx) = self.forward(inputs, states)
        if self.attn_dim == 0:
            return lstm_out, (hx, cx)
        attn_h = torch.tanh(self.attn_proj(lstm_out))
        attn_weights = F.softmax(self.attn_score(attn_h).squeeze(-1), dim=0)
        context = torch.sum(lstm_out * attn_weights.unsqueeze(-1), dim=0, keepdim=True)
        return context, (hx, cx)

class LSTMTagger(nn.Module):
    """
    Sequence tagging model that can switch between classical and quantum LSTM.
    """
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int,
                 tagset_size: int, n_qubits: int = 0,
                 dropout_prob: float = 0.0, attn_dim: int = 0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits,
                              dropout_prob=dropout_prob, attn_dim=attn_dim)
        else:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=0,
                              dropout_prob=dropout_prob, attn_dim=attn_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

    def forward_with_attention(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        context, _ = self.lstm.forward_with_attention(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(context.squeeze(0))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTM", "LSTMTagger"]
