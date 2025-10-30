"""
Hybrid classical LSTM with optional convolution, autoencoder and graph
pre‑processing.

The implementation follows a *combination* scaling paradigm: classical
operations are enriched with lightweight pre‑processing layers that
mirror their quantum counterparts.  The class is fully compatible with
the original QLSTM interface and can be used as a drop‑in
replacement for both the classical and quantum modules.

Key components
--------------
* `Conv`      – 1‑D convolution that emulates a quanvolution filter.
* `Autoencoder` – dimensionality reduction before the LSTM.
* `GraphQNN` – optional graph‑based feature extractor built from
  random weights.
* `QLSTM`     – classical LSTM cell with linear gates (derived from
  the original QLSTM seed).

The module uses only PyTorch and NumPy and is fully self‑contained.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import auxiliary modules from the seed codebase
from.Conv import Conv
from.Autoencoder import Autoencoder
from.GraphQNN import random_network, feedforward
from.QLSTM import QLSTM as ClassicalQLSTM


class HybridQLSTM(nn.Module):
    """
    Hybrid classical LSTM that optionally applies convolution,
    auto‑encoding and graph‑based feature extraction before the LSTM.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        conv_kernel: int = 2,
        autoencoder_latent: int = 32,
        graph_arch: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Optional pre‑processing layers
        self.conv = nn.Conv1d(
            embedding_dim,
            embedding_dim,
            kernel_size=conv_kernel,
            padding=conv_kernel // 2,
        )
        self.autoencoder = Autoencoder(
            input_dim=embedding_dim,
            latent_dim=autoencoder_latent,
        )

        if graph_arch:
            _, self.graph_weights, _, _ = random_network(graph_arch, samples=10)
        else:
            self.graph_weights = None

        # LSTM core: classical linear gates or quantum gates
        if n_qubits > 0:
            self.lstm = ClassicalQLSTM(embedding_dim, hidden_dim, n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _graph_features(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the graph‑based feature extractor."""
        if self.graph_weights is None:
            return x
        h = x
        for w in self.graph_weights:
            h = torch.tanh(w @ h)
        return h

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

    # ------------------------------------------------------------------
    # Core forward pass
    # ------------------------------------------------------------------
    def forward(
        self,
        sentence: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        sentence : torch.Tensor
            LongTensor of shape (seq_len,) containing word indices.

        Returns
        -------
        torch.Tensor
            Log‑softmaxed tag logits of shape (seq_len, tagset_size).
        """
        embeds = self.word_embeddings(sentence)  # (seq_len, emb_dim)

        # 1‑D convolution on the embedding sequence
        conv_out = self.conv(embeds.unsqueeze(0))  # (1, emb_dim, seq_len)
        conv_out = conv_out.squeeze(0).transpose(0, 1)  # (seq_len, emb_dim)

        # Auto‑encoder bottleneck
        ae_out = self.autoencoder(conv_out)

        # Optional graph feature extraction
        graph_out = self._graph_features(ae_out)

        # LSTM step
        lstm_out, _ = self.lstm(graph_out.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


class LSTMTagger(nn.Module):
    """
    Sequence tagging model that uses :class:`HybridQLSTM` or
    the vanilla :class:`nn.LSTM` as the recurrent core.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        conv_kernel: int = 2,
        autoencoder_latent: int = 32,
        graph_arch: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        if n_qubits > 0:
            self.lstm = HybridQLSTM(
                embedding_dim,
                hidden_dim,
                vocab_size,
                tagset_size,
                n_qubits=n_qubits,
                conv_kernel=conv_kernel,
                autoencoder_latent=autoencoder_latent,
                graph_arch=graph_arch,
            )
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        if isinstance(self.lstm, HybridQLSTM):
            # Hybrid forward already includes pre‑processing
            tag_logits = self.lstm(sentence)
        else:
            lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
            tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["HybridQLSTM", "LSTMTagger"]
