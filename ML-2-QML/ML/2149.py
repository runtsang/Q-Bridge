"""
Enhanced classical LSTM with optional dropout, layer‑norm, residuals, and multi‑layer support.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Sequence, Union


class QLSTM(nn.Module):
    """
    Classical LSTM cell that stacks multiple `nn.LSTMCell`s with optional
    dropout, layer‑norm, and residual connections.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input vector.
    hidden_dim : int | Sequence[int]
        Hidden size(s). If a sequence is provided, the LSTM is stacked
        with one layer per element.
    dropout : float, default 0.0
        Dropout probability applied to the output of each layer.
    layernorm : bool, default False
        If True, a `nn.LayerNorm` is inserted after each gate.
    residual : bool, default False
        If True, adds residual connections between consecutive layers.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: Union[int, Sequence[int]],
        dropout: float = 0.0,
        layernorm: bool = False,
        residual: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(hidden_dim, int):
            hidden_dim = (hidden_dim,)
        self.n_layers = len(hidden_dim)
        self.dropout = dropout
        self.residual = residual
        self.layernorm = layernorm

        self.cells = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i, h_dim in enumerate(hidden_dim):
            in_dim = input_dim if i == 0 else hidden_dim[i - 1]
            self.cells.append(nn.LSTMCell(in_dim, h_dim))
            if self.layernorm:
                self.norms.append(nn.LayerNorm(h_dim))
            else:
                self.norms.append(None)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[Tuple[torch.Tensor, torch.Tensor],...] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[Tuple[torch.Tensor, torch.Tensor],...]]:
        """
        Forward pass over a whole sequence.

        Parameters
        ----------
        inputs : Tensor
            Shape `(seq_len, batch, input_dim)`.
        states : Tuple[Tuple[h, c],...] | None
            Optional initial hidden/cell states for each layer.

        Returns
        -------
        outputs : Tensor
            Shape `(seq_len, batch, last_hidden_dim)`.
        new_states : Tuple[Tuple[h, c],...]
            Final hidden/cell states for each layer.
        """
        seq_len, batch_size, _ = inputs.size()
        if states is None:
            states = [
                (
                    torch.zeros(batch_size, cell.hidden_size, device=inputs.device),
                    torch.zeros(batch_size, cell.hidden_size, device=inputs.device),
                )
                for cell in self.cells
            ]
        else:
            assert len(states) == self.n_layers

        h = [s[0] for s in states]
        c = [s[1] for s in states]
        outputs = []

        for t in range(seq_len):
            x = inputs[t]
            for i, cell in enumerate(self.cells):
                h_i, c_i = cell(x, (h[i], c[i]))
                if self.norms[i] is not None:
                    h_i = self.norms[i](h_i)
                if self.residual and i > 0:
                    h_i = h_i + h[i - 1]
                h[i], c[i] = h_i, c_i
                x = h_i
            if self.dropout > 0.0:
                x = F.dropout(x, p=self.dropout, training=self.training)
            outputs.append(x.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)
        new_states = tuple(zip(h, c))
        return outputs, new_states


class LSTMTagger(nn.Module):
    """
    Sequence tagging model that can use the enhanced classical `QLSTM`
    or a standard `nn.LSTM`.

    Parameters
    ----------
    embedding_dim : int
    hidden_dim : int | Sequence[int]
    vocab_size : int
    tagset_size : int
    dropout : float, default 0.0
    layernorm : bool, default False
    residual : bool, default False
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: Union[int, Sequence[int]],
        vocab_size: int,
        tagset_size: int,
        dropout: float = 0.0,
        layernorm: bool = False,
        residual: bool = False,
    ) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTM(
            embedding_dim,
            hidden_dim,
            dropout=dropout,
            layernorm=layernorm,
            residual=residual,
        )
        last_dim = hidden_dim[-1] if isinstance(hidden_dim, (tuple, list)) else hidden_dim
        self.hidden2tag = nn.Linear(last_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning log‑softmax over tag logits.

        Parameters
        ----------
        sentence : Tensor
            Shape `(seq_len, batch)` with word indices.
        """
        embeds = self.word_embeddings(sentence).unsqueeze(1)  # (seq_len, batch, 1, emb)
        lstm_out, _ = self.lstm(embeds.squeeze(2))
        tag_logits = self.hidden2tag(lstm_out.squeeze(1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["QLSTM", "LSTMTagger"]
