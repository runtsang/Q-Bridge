import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class QLSTM(nn.Module):
    """
    Classical LSTM cell with residual skip connections, dropout and layer‑norm.
    The gate matrices are linear and can be optionally stacked with
    layer normalization for improved stability.
    Parameters
    ----------
    input_dim: int
        Dimensionality of input features.
    hidden_dim: int
        Dimensionality of hidden state.
    n_qubits: int
        Kept for API compatibility. Not used in the classical version.
    dropout: float
        Dropout probability applied to gate activations.
    residual: bool
        If True, adds the previous hidden state to the output
        as a residual connection.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0,
                 dropout: float = 0.1, residual: bool = True) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        gate_dim = hidden_dim
        self.forget_linear = nn.Linear(input_dim + hidden_dim, gate_dim, bias=True)
        self.input_linear = nn.Linear(input_dim + hidden_dim, gate_dim, bias=True)
        self.update_linear = nn.Linear(input_dim + hidden_dim, gate_dim, bias=True)
        self.output_linear = nn.Linear(input_dim + hidden_dim, gate_dim, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.residual = residual
        self.layernorm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_linear(combined))
            i = torch.sigmoid(self.input_linear(combined))
            g = torch.tanh(self.update_linear(combined))
            o = torch.sigmoid(self.output_linear(combined))
            cx = f * cx + i * g
            hx_new = o * torch.tanh(cx)
            if self.residual:
                hx_new = hx_new + hx
            hx_new = self.layernorm(hx_new)
            hx_new = self.dropout(hx_new)
            hx = hx_new
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

class LSTMTagger(nn.Module):
    """
    Sequence tagging model that can optionally use the extended QLSTM.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        dropout: float = 0.1,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits,
                              dropout=dropout, residual=residual)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTM", "LSTMTagger"]
