import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm, Dropout
from typing import Tuple


class QLSTMHybrid(nn.Module):
    """
    Classical LSTM cell with expanded encoder and regularization.
    """

    def __init__(self, input_dim: int, hidden_dim: int,
                 n_qubits: int = 0,
                 dropout: float = 0.2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.dropout = Dropout(dropout)

        # Encoder: 1D conv to increase feature dimension
        self.encoder = nn.Conv1d(in_channels=input_dim,
                                 out_channels=hidden_dim,
                                 kernel_size=3,
                                 padding=1,
                                 bias=False)

        self.layer_norm = LayerNorm(2 * hidden_dim)

        # Gates
        self.forget_linear = nn.Linear(2 * hidden_dim, hidden_dim)
        self.input_linear = nn.Linear(2 * hidden_dim, hidden_dim)
        self.update_linear = nn.Linear(2 * hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(self,
                inputs: torch.Tensor,
                states: Tuple[torch.Tensor, torch.Tensor] | None = None
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        inputs: (seq_len, batch, input_dim)
        states: (hx, cx) each (batch, hidden_dim)
        """
        hx, cx = self._init_states(inputs, states)

        # Encode inputs
        seq_len, batch_size, _ = inputs.shape
        x = inputs.permute(1, 2, 0)  # (batch, input_dim, seq_len)
        encoded = self.encoder(x)    # (batch, hidden_dim, seq_len)
        encoded = encoded.permute(2, 0, 1)  # (seq_len, batch, hidden_dim)

        outputs = []
        for x_t in encoded.unbind(dim=0):
            combined = torch.cat([x_t, hx], dim=1)
            combined_ln = self.layer_norm(combined)

            f = torch.sigmoid(self.forget_linear(combined_ln))
            i = torch.sigmoid(self.input_linear(combined_ln))
            g = torch.tanh(self.update_linear(combined_ln))
            o = torch.sigmoid(self.output_linear(combined_ln))

            f = self.dropout(f)
            i = self.dropout(i)
            g = self.dropout(g)
            o = self.dropout(o)

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(self,
                     inputs: torch.Tensor,
                     states: Tuple[torch.Tensor, torch.Tensor] | None
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
    Sequence tagging model using the hybrid LSTM.
    """

    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 n_qubits: int = 0,
                 dropout: float = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTMHybrid(embedding_dim,
                                hidden_dim,
                                n_qubits=n_qubits,
                                dropout=dropout)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTMHybrid", "LSTMTagger"]
