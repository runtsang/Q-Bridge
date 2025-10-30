"""
HybridEstimator module that supports both regression and sequence tagging.
The classical implementation uses a simple feed‑forward regressor or a
standard nn.LSTM for tagging.  The quantum branch is omitted for the
classical module; if use_q is requested an informative error is raised.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridEstimator(nn.Module):
    """Hybrid estimator that can act as a regressor or a sequence tagger.

    Parameters
    ----------
    mode : {"regression", "tagging"}
        Which task to perform.
    use_q : bool, optional
        If True the quantum implementation is used.  In the classical
        module this raises an error because quantum dependencies are not
        available.
    n_qubits : int, optional
        Number of qubits for the quantum implementation.
    embedding_dim : int, optional
        Embedding size for the tagging task.
    hidden_dim : int, optional
        Hidden size for the LSTM.
    vocab_size : int, optional
        Vocabulary size for the embedding layer.
    tagset_size : int, optional
        Number of tags for the tagging task.
    """

    def __init__(
        self,
        mode: str = "regression",
        use_q: bool = False,
        n_qubits: int = 0,
        embedding_dim: int = 50,
        hidden_dim: int = 100,
        vocab_size: int = 10000,
        tagset_size: int = 10,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.use_q = use_q
        self.n_qubits = n_qubits

        if self.use_q:
            raise RuntimeError(
                "Quantum dependencies are not available in the classical module. "
                "Use the qml implementation instead."
            )

        if mode == "regression":
            self.regressor = _ClassicalRegressor()
        elif mode == "tagging":
            self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
            self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        else:
            raise ValueError(f"Unsupported mode {mode!r}")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.mode == "regression":
            return self.regressor(inputs)
        else:
            # tagging mode
            embeds = self.word_embeddings(inputs)
            lstm_out, _ = self.lstm(embeds)
            tag_logits = self.hidden2tag(lstm_out)
            return F.log_softmax(tag_logits, dim=-1)

class _ClassicalRegressor(nn.Module):
    """Simple fully‑connected regressor used for the regression mode."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def EstimatorQNN() -> HybridEstimator:
    """Factory that returns a classical HybridEstimator instance."""
    return HybridEstimator()

__all__ = ["HybridEstimator", "EstimatorQNN"]
