import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class QLSTMEnhanced(nn.Module):
    """
    Hybrid classical LSTM cell with enhanced gating.

    Features
    --------
    * Shared linear that outputs all four gates in one pass.
    * Optional layer‑norm per gate for stable training.
    * Drop‑out applied after each gate to regularise the gate
      activations.
    * Residual connection between the hidden state and the gate
      output, allowing gradients to flow straight through the cell.
    * Compatibility with the original QLSTM interface: ``(input_dim,
      hidden_dim, n_qubits=0)``.  The ``n_qubits`` argument is kept
      for API compatibility but is ignored in the classical
      implementation.

    This class can be dropped into the original ``LSTMTagger`` or
    any other architecture that expects an ``nn.Module`` with a
    ``forward`` method returning ``(output, (hx, cx))``.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int = 0,
        dropout: float = 0.1,
        use_layernorm: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.use_layernorm = use_layernorm

        # Single linear mapping for all gates
        self.gate_linear = nn.Linear(input_dim + hidden_dim, 4 * hidden_dim, bias=True)

        # Optional per‑gate layer‑norm
        if use_layernorm:
            self.ln_f = nn.LayerNorm(hidden_dim)
            self.ln_i = nn.LayerNorm(hidden_dim)
            self.ln_g = nn.LayerNorm(hidden_dim)
            self.ln_o = nn.LayerNorm(hidden_dim)
        else:
            self.ln_f = self.ln_i = self.ln_g = self.ln_o = lambda x: x

        self.dropout_layer = nn.Dropout(dropout)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []

        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            gates = self.gate_linear(combined)
            f, i, g, o = gates.chunk(4, dim=1)

            # Per‑gate normalisation and activation
            f = torch.sigmoid(self.ln_f(f))
            i = torch.sigmoid(self.ln_i(i))
            g = torch.tanh(self.ln_g(g))
            o = torch.sigmoid(self.ln_o(o))

            # Drop‑out on gate activations
            f = self.dropout_layer(f)
            i = self.dropout_layer(i)
            g = self.dropout_layer(g)
            o = self.dropout_layer(o)

            # Classical LSTM update
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)

            # Residual connection to aid gradient flow
            hx = hx + self.dropout_layer(hx)

            outputs.append(hx.unsqueeze(0))

        output_tensor = torch.cat(outputs, dim=0)
        return output_tensor, (hx, cx)

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


class LSTMTaggerEnhanced(nn.Module):
    """
    Sequence‑tagging model that can switch between the enhanced
    classical LSTM cell and the standard ``nn.LSTM``.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        dropout: float = 0.1,
        use_layernorm: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTMEnhanced(
                embedding_dim,
                hidden_dim,
                n_qubits=n_qubits,
                dropout=dropout,
                use_layernorm=use_layernorm,
            )
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["QLSTMEnhanced", "LSTMTaggerEnhanced"]
