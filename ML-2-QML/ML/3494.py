"""Classical LSTM module with optional quantum-inspired gating and a sampler submodule.

This module mirrors the interfaces of the original QLSTM but
provides a fully classical implementation that can be used as a
drop‑in replacement.  A lightweight sampler network is included
to illustrate how a simple classical network can be coupled
to the LSTM gates.  The design allows the same class name to be
re‑used in a quantum variant, enabling direct comparison.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class SamplerModule(nn.Module):
    """Simple two‑layer linear sampler producing a probability distribution.

    The network maps a 2‑dimensional input to a 2‑dimensional
    soft‑max output.  It is used only when `use_sampler` is True.
    """
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return F.softmax(self.net(inputs), dim=-1)

class QLSTMGen113(nn.Module):
    """Classical LSTM with optional quantum‑style gates.

    Parameters
    ----------
    input_dim : int
        Size of each input vector.
    hidden_dim : int
        Size of the hidden state.
    n_qubits : int, optional
        Number of qubits that would be used in the quantum variant.
        In the classical implementation this argument is purely
        informative and does not change the behaviour.
    use_sampler : bool, optional
        If ``True``, a small sampler network is applied to the
        input before it is fed to the gates.
    """
    def __init__(self, input_dim: int, hidden_dim: int,
                 n_qubits: int = 0, use_sampler: bool = False) -> None:
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
            self.sampler = SamplerModule()
        else:
            self.sampler = None

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Run a forward pass through the LSTM.

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape ``(seq_len, batch, input_dim)``.
        states : tuple or None
            Previous hidden and cell states.  If ``None`` they are
            initialized to zero tensors.

        Returns
        -------
        outputs : torch.Tensor
            Tensor of shape ``(seq_len, batch, hidden_dim)`` containing
            the hidden state at each time step.
        final_state : tuple
            Final hidden and cell state.
        """
        hx, cx = self._init_states(inputs, states)
        outputs = []

        for x in inputs.unbind(dim=0):
            # Pre‑process with sampler if requested
            if self.sampler is not None:
                # sampler expects 2‑dim input; we project x to 2 dims
                # by linear projection (simple mean of first two dims)
                # This is a placeholder; in practice you may choose a
                # more meaningful mapping.
                proj = x[:, :2] if x.shape[1] >= 2 else F.pad(x, (0, 2 - x.shape[1]))
                samp = self.sampler(proj)
                # broadcast to match hidden dimension
                samp_exp = samp.unsqueeze(-1).expand(-1, self.hidden_dim)
                # Modulate the input before gating
                x = x * samp_exp

            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_linear(combined))
            i = torch.sigmoid(self.input_linear(combined))
            g = torch.tanh(self.update_linear(combined))
            o = torch.sigmoid(self.output_linear(combined))

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))

        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

class LSTMTagger(nn.Module):
    """Sequence tagging model that can use either the classical
    :class:`QLSTMGen113` or a standard :class:`torch.nn.LSTM`."""
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
        if n_qubits > 0 or use_sampler:
            self.lstm = QLSTMGen113(embedding_dim, hidden_dim,
                                    n_qubits=n_qubits,
                                    use_sampler=use_sampler)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTMGen113", "LSTMTagger"]
