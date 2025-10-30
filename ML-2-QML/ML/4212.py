"""Hybrid fraud detection model combining photonic-inspired feature extraction,
quantum-enhanced LSTM gating and a softmax sampler.

The model is a drop‑in replacement for the classical FraudDetection program
and can be trained with standard PyTorch pipelines.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from FraudDetection import FraudLayerParameters, build_fraud_detection_program
from QLSTM import QLSTM

class SamplerQNN(nn.Module):
    """Simple softmax classifier mimicking a quantum sampler network."""
    def __init__(self, input_dim: int = 2, hidden_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(inputs), dim=-1)

class FraudDetectionHybrid(nn.Module):
    """Hybrid fraud detection model.

    Architecture:
        1. Photonic-inspired feature extractor built from FraudLayerParameters.
        2. Quantum‑enhanced LSTM (QLSTM) to capture sequential dependencies.
        3. SamplerQNN classifier for final fraud probability.
    """
    def __init__(self,
                 input_params: FraudLayerParameters,
                 layer_params: list[FraudLayerParameters],
                 lstm_hidden_dim: int = 32,
                 n_qubits: int = 8):
        super().__init__()
        # Feature extractor
        self.feature_extractor = build_fraud_detection_program(input_params, layer_params)

        # LSTM
        self.lstm = QLSTM(input_dim=2, hidden_dim=lstm_hidden_dim, n_qubits=n_qubits)

        # Classifier
        self.classifier = SamplerQNN(input_dim=lstm_hidden_dim, hidden_dim=16)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs shape (batch, seq_len, 2)
        batch, seq_len, _ = inputs.shape
        # Flatten batch and seq for feature extractor
        flat = inputs.view(batch * seq_len, -1)
        features = self.feature_extractor(flat)  # (batch*seq_len, 2)
        # Reshape back to sequence
        seq_features = features.view(batch, seq_len, -1)
        # LSTM expects (seq_len, batch, input_dim)
        lstm_in = seq_features.transpose(0, 1)  # (seq_len, batch, 2)
        lstm_out, _ = self.lstm(lstm_in)
        # Take last hidden state
        final = lstm_out[-1]  # (batch, hidden_dim)
        logits = self.classifier(final)
        return logits
