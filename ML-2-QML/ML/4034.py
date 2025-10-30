"""Hybrid classical-quantum LSTM with kernelised gates.

This module defines :class:`HybridQLSTM` that replaces the linear
gates of a standard LSTM with RBF‑kernel based feature maps.
The kernel centres are learnable parameters and the kernels are
aggregated through a small linear layer before applying the usual
sigmoid/tanh activations.  The implementation remains fully
PyTorch‑based and can be dropped in wherever :class:`QLSTM` or
:class:`nn.LSTM` was used.

The design mirrors the quantum implementation below but stays
completely classical, enabling easy ablation studies and
benchmarking against the quantum version.

Author: gpt-oss-20b
"""

from __future__ import annotations

from typing import Tuple, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class HybridQLSTM(nn.Module):
    """Classical LSTM cell with kernelised gates.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input at each time step.
    hidden_dim : int
        Dimensionality of the hidden state.
    num_kernels : int, default=32
        Number of kernel centres per gate.  Each gate learns
        `num_kernels` centres which are used to build a feature
        representation of the combined input/hidden vector.
    gamma : float, default=1.0
        RBF kernel width.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_kernels: int = 32,
        gamma: float = 1.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_kernels = num_kernels
        self.gamma = gamma

        # Kernel centres for each gate: shape (num_kernels, input_dim+hidden_dim)
        init_center = torch.randn(num_kernels, input_dim + hidden_dim) * 0.1
        self.kernel_centers_f = nn.Parameter(init_center.clone())
        self.kernel_centers_i = nn.Parameter(init_center.clone())
        self.kernel_centers_g = nn.Parameter(init_center.clone())
        self.kernel_centers_o = nn.Parameter(init_center.clone())

        # Linear maps from kernel features to gate space
        self.linear_f = nn.Linear(num_kernels, hidden_dim)
        self.linear_i = nn.Linear(num_kernels, hidden_dim)
        self.linear_g = nn.Linear(num_kernels, hidden_dim)
        self.linear_o = nn.Linear(num_kernels, hidden_dim)

    @staticmethod
    def _rbf_kernel(x: torch.Tensor, centres: torch.Tensor, gamma: float) -> torch.Tensor:
        """Compute RBF kernel between each row of ``x`` and ``centres``."""
        # x: (B, D), centres: (K, D)
        diff = x.unsqueeze(1) - centres.unsqueeze(0)  # (B, K, D)
        return torch.exp(-gamma * (diff.pow(2).sum(dim=2)))  # (B, K)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Run the kernelised LSTM over a sequence.

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape (T, B, I) where ``I`` is ``input_dim``.
        states : tuple, optional
            ``(h_0, c_0)`` initial hidden and cell states.
        """
        h, c = self._init_states(inputs, states)
        outputs = []

        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, h], dim=1)  # (B, I+H)
            # Kernel features
            k_f = self._rbf_kernel(combined, self.kernel_centers_f, self.gamma)
            k_i = self._rbf_kernel(combined, self.kernel_centers_i, self.gamma)
            k_g = self._rbf_kernel(combined, self.kernel_centers_g, self.gamma)
            k_o = self._rbf_kernel(combined, self.kernel_centers_o, self.gamma)

            # Gate activations
            f = torch.sigmoid(self.linear_f(k_f))
            i = torch.sigmoid(self.linear_i(k_i))
            g = torch.tanh(self.linear_g(k_g))
            o = torch.sigmoid(self.linear_o(k_o))

            c = f * c + i * g
            h = o * torch.tanh(c)
            outputs.append(h.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)  # (T, B, H)
        return outputs, (h, c)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, device=device)
        return h, c


class LSTMTagger(nn.Module):
    """Sequence tagging model that can use either the classical HybridQLSTM
    or a standard ``nn.LSTM`` for experimentation.

    Parameters
    ----------
    embedding_dim : int
        Embedding dimensionality of input tokens.
    hidden_dim : int
        Hidden state size of the LSTM.
    vocab_size : int
        Size of the token vocabulary.
    tagset_size : int
        Number of output tags.
    use_kernel : bool, default=False
        If ``True``, use :class:`HybridQLSTM` with kernelised gates;
        otherwise fall back to :class:`nn.LSTM`.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        use_kernel: bool = False,
        num_kernels: int = 32,
        gamma: float = 1.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if use_kernel:
            self.lstm = HybridQLSTM(
                embedding_dim,
                hidden_dim,
                num_kernels=num_kernels,
                gamma=gamma,
            )
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        sentence : torch.Tensor
            Tensor of shape (T, B) containing token indices.
        """
        embeds = self.word_embeddings(sentence)  # (T, B, E)
        if isinstance(self.lstm, nn.LSTM):
            lstm_out, _ = self.lstm(embeds)
        else:
            lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out)  # (T, B, tagset)
        return F.log_softmax(tag_logits, dim=2)


__all__ = ["HybridQLSTM", "LSTMTagger"]
