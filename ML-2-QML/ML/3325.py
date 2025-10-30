"""Hybrid classical LSTM with optional sampler module for gate approximation.

This module extends the original pure‑PyTorch QLSTM by adding a lightweight
`SamplerModule` that emulates the behaviour of the quantum sampler used in
the QML version.  The `QLSTM` class can operate in two modes:
  * Classical gates (default) – identical to the original implementation.
  * Sampler‑based gates – each gate is replaced by a small neural network
    that outputs a probability distribution; the first component is used
    as the gate value.

The `LSTMTagger` remains a drop‑in replacement for the original model
and exposes the same interface.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SamplerModule(nn.Module):
    """
    Classical approximation of the quantum sampler in the QML branch.
    Produces a two‑element probability vector via a small feed‑forward
    network.  The first element is used as the gate activation.
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape (batch, 2) containing two input features.

        Returns
        -------
        torch.Tensor
            Softmax‑normalised probabilities of shape (batch, 2).
        """
        return F.softmax(self.net(inputs), dim=-1)


class QLSTM(nn.Module):
    """
    Classical LSTM cell with optional sampler‑based gates.

    Parameters
    ----------
    input_dim : int
        Dimensionality of input features.
    hidden_dim : int
        Dimensionality of hidden state.
    n_qubits : int
        Number of qubits for the quantum branch; ignored in classical mode.
    use_sampler : bool
        If True, each gate is replaced by a `SamplerModule` that outputs
        probabilities; otherwise the gate is a standard sigmoid/tanh.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int,
        use_sampler: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.use_sampler = use_sampler

        gate_dim = hidden_dim
        self.forget_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, gate_dim)

        if self.use_sampler:
            # Replace each gate with a sampler network; only first output
            # component is used as gate value.
            self.forget_sampler = SamplerModule()
            self.input_sampler = SamplerModule()
            self.update_sampler = SamplerModule()
            self.output_sampler = SamplerModule()

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:  # type: ignore[override]
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_linear(combined))
            i = torch.sigmoid(self.input_linear(combined))
            g = torch.tanh(self.update_linear(combined))
            o = torch.sigmoid(self.output_linear(combined))
            if self.use_sampler:
                # Convert linear outputs to two‑dim vectors and sample
                f = self.forget_sampler(f).select(1, 0)
                i = self.input_sampler(i).select(1, 0)
                g = self.update_sampler(g).select(1, 0)
                o = self.output_sampler(o).select(1, 0)
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
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
    Sequence tagging model that can switch between a classical LSTM and
    a sampler‑based LSTM.

    Parameters
    ----------
    embedding_dim : int
        Dimensionality of word embeddings.
    hidden_dim : int
        Hidden state size.
    vocab_size : int
        Size of the vocabulary.
    tagset_size : int
        Number of output tags.
    n_qubits : int
        Number of qubits for quantum mode (ignored in classical mode).
    use_sampler : bool
        If True, use the sampler‑based LSTM; otherwise use the classical LSTM.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        use_sampler: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits, use_sampler=use_sampler)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["QLSTM", "LSTMTagger", "SamplerModule"]
