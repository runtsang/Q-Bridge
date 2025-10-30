import torch
from torch import nn
import torch.nn.functional as F

# Local imports – ensure the modules are in the same package
from.Quanvolution import QuanvolutionFilter
from.Autoencoder import AutoencoderNet, AutoencoderConfig
from.QLSTM import QLSTM

class EstimatorQNNHybrid(nn.Module):
    """
    Classical hybrid model that combines a Quanvolution filter,
    an auto‑encoder bottleneck, and a classical LSTM before a final
    linear classifier.  It mirrors the structure of the original
    EstimatorQNN but adds feature extraction and sequence modelling
    to improve representation learning.
    """
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.quanvolution = QuanvolutionFilter()
        # Autoencoder encoder part only
        config = AutoencoderConfig(
            input_dim=4 * 14 * 14,
            latent_dim=32,
            hidden_dims=(128, 64),
            dropout=0.1,
        )
        self.autoencoder = AutoencoderNet(config)
        self.lstm = QLSTM(input_dim=config.latent_dim, hidden_dim=64, n_qubits=4)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 1, 28, 28)
        x = self.quanvolution(x)                      # (batch, 4*14*14)
        x = self.autoencoder.encode(x)                # (batch, latent_dim)
        # LSTM expects sequence dimension first
        seq = x.unsqueeze(0)                          # (1, batch, latent_dim)
        lstm_out, _ = self.lstm(seq)                  # (1, batch, hidden_dim)
        out = self.classifier(lstm_out.squeeze(0))    # (batch, num_classes)
        return out

__all__ = ["EstimatorQNNHybrid"]
