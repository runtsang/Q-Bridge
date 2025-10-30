import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class QLSTM(nn.Module):
    """Classical LSTM cell augmented with stochastic gate dropout and a lightweight
    auto‑encoder for the hidden state.  The architecture remains drop‑in compatible
    with the original QLSTM while providing richer regularisation and a
    per‑time‑step hybrid mode that can be selected via the ``mode`` argument.
    """

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.dropout = dropout

        # Classical linear gates
        self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # Auto‑encoder for hidden state (helps with state‑dropout)
        self.encoder = nn.Linear(hidden_dim, hidden_dim // 2)
        self.decoder = nn.Linear(hidden_dim // 2, hidden_dim)

        # Optional self‑attention module for context‑aware gating
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=1, batch_first=True)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        mode: str = "classical",
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Parameters
        ----------
        inputs : (seq_len, batch, input_dim)
            Input sequence.
        states : (h, c) optional
            Initial hidden and cell states.
        mode : {"classical", "quantum", "hybrid"}
            * classical – use only classical gates.
            * quantum   – use only quantum‑derived gates (via a stub that
              mimics the behaviour of the quantum circuit).
            * hybrid    – alternate between classical and quantum gates at
              each time step.
        """
        hx, cx = self._init_states(inputs, states)
        outputs = []

        for t, x in enumerate(inputs.unbind(dim=0)):
            combined = torch.cat([x, hx], dim=1)

            # Gate generation
            if mode == "classical":
                f = torch.sigmoid(self.forget_linear(combined))
                i = torch.sigmoid(self.input_linear(combined))
                g = torch.tanh(self.update_linear(combined))
                o = torch.sigmoid(self.output_linear(combined))
            elif mode == "quantum":
                # A lightweight “quantum” stub: the same linear layers but
                # followed by a random sign flip to emulate measurement noise.
                f = torch.sigmoid(self.forget_linear(combined))
                i = torch.sigmoid(self.input_linear(combined))
                g = torch.tanh(self.update_linear(combined))
                o = torch.sigmoid(self.output_linear(combined))
                noise = torch.randn_like(f) * 0.05
                f, i, g, o = f + noise, i + noise, (g + noise) / 2, (o + noise) / 2
            else:  # hybrid
                if t % 2 == 0:
                    f = torch.sigmoid(self.forget_linear(combined))
                    i = torch.sigmoid(self.input_linear(combined))
                    g = torch.tanh(self.update_linear(combined))
                    o = torch.sigmoid(self.output_linear(combined))
                else:
                    f = torch.sigmoid(self.forget_linear(combined))
                    i = torch.sigmoid(self.input_linear(combined))
                    g = torch.tanh(self.update_linear(combined))
                    o = torch.sigmoid(self.output_linear(combined))
                    noise = torch.randn_like(f) * 0.05
                    f, i, g, o = f + noise, i + noise, (g + noise) / 2, (o + noise) / 2

            # State‑dropout via auto‑encoder
            hx_enc = self.encoder(hx)
            hx_dec = self.decoder(hx_enc)
            hx = hx_dec + (1 - self.dropout) * hx

            # Attention‑based context gating (optional)
            attn_out, _ = self.attn(hx.unsqueeze(0), hx.unsqueeze(0), hx.unsqueeze(0))
            hx = hx + attn_out.squeeze(0)

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)

            outputs.append(hx.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

class LSTMTagger(nn.Module):
    """Sequence tagging model that uses the enhanced QLSTM cell."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits, dropout=dropout)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        return F.log_softmax(self.hidden2tag(lstm_out.view(len(sentence), -1)), dim=1)

__all__ = ["QLSTM", "LSTMTagger"]
