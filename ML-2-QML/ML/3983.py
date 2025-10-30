"""Enhanced classical LSTM tagger with parameter sweep and shot‑noise simulation.

The class re‑implements the original `QLSTMTagger` but augments it with
two new features:

* :py:meth:`evaluate` – evaluates the network over a grid of
  embedding‑weight vectors.  The method mimics the behaviour of
  ``FastEstimator`` from the reference pair: it accepts a list of
  parameter sets and an optional shot‑noise configuration.

* :py:meth:`add_noise` – injects Gaussian noise into the logits,
  providing a quick way to simulate stochastic inference.

The implementation keeps the original public API (``forward``,
``hidden2tag``) so it can be swapped in place of the anchor file.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Iterable, Sequence
from typing import Callable, List, Tuple

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    t = torch.as_tensor(values, dtype=torch.float32)
    if t.ndim == 1:
        t = t.unsqueeze(0)
    return t

class QLSTMTagger(nn.Module):
    """
    Classical LSTM tagger with optional parameter‑sweep evaluation.

    Parameters
    ----------
    embedding_dim : int
        Dimensionality of token embeddings.
    hidden_dim : int
        Size of LSTM hidden state.
    vocab_size : int
        Size of the vocabulary.
    tagset_size : int
        Number of output tags.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a single sentence.

        Parameters
        ----------
        sentence : torch.Tensor
            LongTensor of shape ``(seq_len,)`` containing token indices.

        Returns
        -------
        torch.Tensor
            Log‑softmax logits of shape ``(seq_len, tagset_size)``.
        """
        embeds = self.word_embeddings(sentence).unsqueeze(0)  # (1, seq_len, embed)
        lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out.squeeze(0))
        return F.log_softmax(tag_logits, dim=1)

    # ------------------------------------------------------------------
    #  Evaluation utilities (FastEstimator inspired)
    # ------------------------------------------------------------------
    def _apply_noise(
        self,
        logits: torch.Tensor,
        shots: int | None = None,
        seed: int | None = None,
    ) -> torch.Tensor:
        """Optionally inject Gaussian noise to simulate stochastic inference."""
        if shots is None:
            return logits
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, 1.0 / np.sqrt(shots), size=logits.shape)
        return logits + torch.from_numpy(noise).to(logits.device)

    def evaluate(
        self,
        sentences: Sequence[torch.Tensor],
        parameter_sets: Sequence[Sequence[float]] | None = None,
        shots: int | None = None,
        seed: int | None = None,
        observables: Iterable[ScalarObservable] | None = None,
    ) -> List[List[float]]:
        """
        Evaluate the model over a list of sentences and optional
        parameter sets.  When *parameter_sets* is provided, the
        embedding matrix is temporarily replaced by the given weight
        vectors.

        Parameters
        ----------
        sentences : Sequence[torch.Tensor]
            Iterable of 1‑D LongTensors each of shape ``(seq_len,)``.
        parameter_sets : Sequence[Sequence[float]] | None, default None
            Each inner sequence must match the size of the embedding
            matrix (``vocab_size * embedding_dim``).  If provided, the
            embeddings are overridden before evaluation.
        shots : int | None, default None
            If set, Gaussian noise is added to logits to emulate
            shot‑level sampling.
        seed : int | None, default None
            Random seed for reproducibility of the noise.
        observables : Iterable[ScalarObservable] | None, default None
            Functions that map the logits tensor to a scalar.  If not
            supplied, the mean of the logits is returned.

        Returns
        -------
        List[List[float]]
            Outer list corresponds to sentences, inner list to
            observables.
        """
        # Preserve original embedding weights
        original_weights = self.word_embeddings.weight.data.clone()

        results: List[List[float]] = []
        self.eval()
        with torch.no_grad():
            for sentence in sentences:
                if parameter_sets is not None:
                    # Replace embeddings
                    new_weights = torch.tensor(
                        np.concatenate(parameter_sets),
                        dtype=torch.float32,
                    ).view_as(original_weights)
                    self.word_embeddings.weight.data = new_weights

                logits = self.forward(sentence)
                logits = self._apply_noise(logits, shots, seed)

                if observables is None:
                    obs_values = [float(logits.mean().item())]
                else:
                    obs_values = [float(obs(logits).mean().item()) if isinstance(obs(logits), torch.Tensor)
                                  else float(obs(logits)) for obs in observables]

                results.append(obs_values)

        # Restore original weights
        self.word_embeddings.weight.data = original_weights
        return results

__all__ = ["QLSTMTagger"]
