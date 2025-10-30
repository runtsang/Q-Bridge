"""Hybrid LSTM model with optional classical convolutional front‑end.

The classical implementation mirrors the structure of the quantum counterpart but
uses standard PyTorch layers.  The wrapper class ``HybridQLSTM`` can instantiate
either a sequence‑tagging model or an image classifier depending on the arguments.
The design keeps the API identical to the quantum module, enabling a seamless
switch between back‑ends.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuanvolutionFilter(nn.Module):
    """Classical 2×2 convolutional filter (4 output channels)."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)


class QuanvolutionClassifier(nn.Module):
    """Classifier that uses the classical quanvolution filter."""
    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)


class QLSTM(nn.Module):
    """Classical LSTM with linear gates."""
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.lstm(inputs)
        return outputs


class LSTMTagger(nn.Module):
    """Sequence tagging with a classical LSTM."""
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # In the classical module we ignore n_qubits and use a plain LSTM
        self.lstm = QLSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


class HybridQLSTM(nn.Module):
    """Unified classical API mirroring the quantum module."""
    def __init__(self, mode: str = "classical", **kwargs) -> None:
        super().__init__()
        if mode not in {"classical", "quantum"}:
            raise ValueError("mode must be 'classical' or 'quantum'")
        self.mode = mode
        # The classical wrapper always uses the plain LSTMTagger.
        self.model = LSTMTagger(**kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


__all__ = [
    "QuanvolutionFilter",
    "QuanvolutionClassifier",
    "QLSTM",
    "LSTMTagger",
    "HybridQLSTM",
]
