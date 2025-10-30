"""QuantumNATEnhanced – a classical research asset combining CNN, regression, LSTM and sampler."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["QuantumNATEnhanced"]


class QuantumNATEnhanced(nn.Module):
    """
    Classical implementation that bundles together:

    * 2‑D CNN + fully connected projection (from Quantum‑NAT).
    * Linear regression head over the compressed feature vector.
    * LSTM tagger for sequence data.
    * Sampler network that outputs a probability distribution over a 2‑dimensional space.

    The class exposes four convenience methods that mirror the distinct data modalities
    used in the seed projects.  Each method returns the appropriate tensor shape and
    can be composed into larger pipelines.
    """

    def __init__(self, n_lstm_layers: int = 1, lstm_hidden: int = 32) -> None:
        super().__init__()
        # CNN backbone
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Fully‑connected projection to 4‑dimensional embedding
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
        self.norm = nn.BatchNorm1d(4)

        # Regression head (mirrors QModel)
        self.regressor = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

        # LSTM tagger (mirrors LSTMTagger)
        self.lstm = nn.LSTM(
            input_size=4,
            hidden_size=lstm_hidden,
            num_layers=n_lstm_layers,
            batch_first=True,
        )
        self.hidden2tag = nn.Linear(lstm_hidden, 10)  # arbitrary tagset size

        # Sampler network
        self.sampler = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    # ------------------------------------------------------------------
    # Image processing pipeline
    # ------------------------------------------------------------------
    def forward_image(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (bsz, 1, 28, 28) grayscale image batch
        :return: (bsz, 4) embedding
        """
        feats = self.features(x)
        flat = feats.view(feats.shape[0], -1)
        out = self.fc(flat)
        return self.norm(out)

    # ------------------------------------------------------------------
    # Regression pipeline
    # ------------------------------------------------------------------
    def forward_regression(self, features: torch.Tensor) -> torch.Tensor:
        """
        :param features: (bsz, 4) embedding from `forward_image`
        :return: (bsz,) scalar predictions
        """
        return self.regressor(features).squeeze(-1)

    # ------------------------------------------------------------------
    # Sequence tagging pipeline
    # ------------------------------------------------------------------
    def forward_sequence(self, seq: torch.Tensor) -> torch.Tensor:
        """
        :param seq: (bsz, seq_len, 4) sequence of embeddings
        :return: (bsz, seq_len, tagset_size)
        """
        lstm_out, _ = self.lstm(seq)
        logits = self.hidden2tag(lstm_out)
        return F.log_softmax(logits, dim=-1)

    # ------------------------------------------------------------------
    # Sampler forward
    # ------------------------------------------------------------------
    def sample(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        :param inputs: (bsz, 2) raw inputs
        :return: (bsz, 2) softmax probabilities
        """
        return F.softmax(self.sampler(inputs), dim=-1)
