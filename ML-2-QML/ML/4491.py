"""Hybrid classical LSTM model with optional quantum features and efficient evaluation.

This module extends the original QLSTM by adding:
* a lightweight convolutional feature extractor (ConvFilter)
* a FastEstimator wrapper that adds optional shot noise
* a classical feed‑forward classifier head built via `build_classifier_circuit`
* a tagger interface that mimics the original `LSTMTagger` API.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

# Local imports from the reference pairs
from.FastBaseEstimator import FastEstimator
from.Conv import Conv as ConvFactory
from.QuantumClassifierModel import build_classifier_circuit


class HybridQLSTM(nn.Module):
    """Classical hybrid LSTM that can optionally use a quantum LSTM gate
    implementation when ``n_qubits > 0``.  The quantum path is a no‑op
    placeholder; the real quantum variant lives in :mod:`qlm.qml_code`."""
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int = 0,
        conv_kernel: int = 2,
        classifier_depth: int = 2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Feature extractor
        self.conv = ConvFactory(kernel_size=conv_kernel)

        # LSTM backbone
        if n_qubits > 0:
            # Placeholder for quantum LSTM; falls back to classical
            self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        else:
            self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

        # Classifier head
        self.classifier, _, _, _ = build_classifier_circuit(hidden_dim, classifier_depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that applies the convolutional feature extractor,
        feeds the result to the LSTM, and finally maps hidden states
        to class logits via the classifier head.
        """
        # Assume input shape (batch, seq_len, features)
        batch, seq_len, _ = x.shape
        # Flatten sequence and apply conv on each feature vector
        x_flat = x.view(batch * seq_len, -1)
        # Convert to numpy for conv.run
        conv_inputs = x_flat.cpu().numpy().reshape(-1, self.conv.kernel_size, self.conv.kernel_size)
        conv_outputs = [self.conv.run(sample) for sample in conv_inputs]
        conv_tensor = torch.tensor(conv_outputs, device=x.device).float()
        conv_tensor = conv_tensor.view(batch, seq_len, -1)

        lstm_out, _ = self.lstm(conv_tensor)
        logits = self.classifier(lstm_out)
        return logits

    def evaluate(
        self,
        observables: Iterable[callable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """
        Wrapper around :class:`FastEstimator` that adds Gaussian shot noise
        to the deterministic predictions.
        """
        estimator = FastEstimator(self)
        return estimator.evaluate(observables, parameter_sets, shots=1000)


class HybridTagger(nn.Module):
    """Sequence tagging wrapper that mirrors the original LSTMTagger API."""
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
    ) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.model = HybridQLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)

        # Linear layer to map hidden state to tag logits
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        logits = self.model(embeds)
        tag_logits = self.hidden2tag(logits)
        return F.log_softmax(tag_logits, dim=-1)


__all__ = ["HybridQLSTM", "HybridTagger"]
