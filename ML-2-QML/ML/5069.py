"""Hybrid classical LSTM with optional CNN feature extractor and fast estimator."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Iterable, Callable, List, Sequence, Optional

class HybridQLSTM(nn.Module):
    """Classical hybrid model inspired by QLSTM and Quantum‑NAT.

    Parameters
    ----------
    input_dim : int
        Dimension of input embeddings.
    hidden_dim : int
        Hidden size of the LSTM.
    vocab_size : int
        Vocabulary size for embeddings.
    tagset_size : int
        Number of tags for sequence tagging.
    use_cnn : bool, default False
        Whether to prepend a simple 1‑D CNN feature extractor.
    cnn_features : nn.Module, optional
        Custom CNN module; if None a lightweight default is used.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 use_cnn: bool = False,
                 cnn_features: nn.Module | None = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, input_dim)
        self.use_cnn = use_cnn
        if use_cnn:
            self.cnn = cnn_features or nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(2),
            )
        else:
            self.cnn = None
        lstm_input_dim = hidden_dim if use_cnn else input_dim
        self.lstm = nn.LSTM(lstm_input_dim, hidden_dim, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """Sequence tagging forward pass."""
        embeds = self.word_embeddings(sentence)
        if self.cnn is not None:
            # Conv1d expects (batch, channels, seq_len)
            embeds = embeds.permute(0, 2, 1)
            embeds = self.cnn(embeds)
            embeds = embeds.permute(0, 2, 1)
        lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=-1)

    # ------------------------------------------------------------------
    # Fast estimation utilities – borrowed from FastBaseEstimator
    # ------------------------------------------------------------------
    @staticmethod
    def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
        tensor = torch.as_tensor(values, dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    def evaluate(self,
                 observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
                 parameter_sets: Sequence[Sequence[float]]) -> List[List[float]]:
        """Evaluate the model on a batch of parameter sets.

        The model is evaluated in eval mode without gradients.  Observables
        are callables that map the model output to a scalar.  Gaussian
        shot noise can be emulated by wrapping this method in
        :class:`FastEstimator`.
        """
        if not observables:
            observables = [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []
        self.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inp = self._ensure_batch(params)
                out = self(inp)
                row: List[float] = []
                for obs in observables:
                    val = obs(out)
                    if isinstance(val, torch.Tensor):
                        val = val.mean().item()
                    row.append(val)
                results.append(row)
        return results

    # ------------------------------------------------------------------
    # Convenience estimator network that mirrors EstimatorQNN
    # ------------------------------------------------------------------
    @staticmethod
    def EstimatorQNN() -> nn.Module:
        """Return a lightweight fully‑connected regressor."""
        class EstimatorNN(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(2, 8),
                    nn.Tanh(),
                    nn.Linear(8, 4),
                    nn.Tanh(),
                    nn.Linear(4, 1),
                )

            def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
                return self.net(inputs)

        return EstimatorNN()
