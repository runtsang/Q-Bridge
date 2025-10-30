import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridNatModel(nn.Module):
    """Classical Nat model combining CNN, autoencoder, and LSTM."""
    def __init__(
        self,
        conv_channels=(8, 16),
        latent_dim=32,
        hidden_dim=64,
        lstm_layers=1,
    ) -> None:
        super().__init__()
        # Convolutional feature extractor
        self.encoder = nn.Sequential(
            nn.Conv2d(1, conv_channels[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Autoencoder bottleneck
        flat_dim = conv_channels[1] * 7 * 7
        self.auto_enc = nn.Sequential(
            nn.Linear(flat_dim, latent_dim),
            nn.ReLU(),
        )
        self.auto_dec = nn.Sequential(
            nn.Linear(latent_dim, flat_dim),
            nn.ReLU(),
        )
        # Classical LSTM for sequence modeling
        self.lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
        )
        # Final classifier
        self.classifier = nn.Linear(hidden_dim, 4)
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature extraction
        feats = self.encoder(x)
        flat = feats.view(feats.size(0), -1)
        # Autoencoder bottleneck
        z = self.auto_enc(flat)
        # Sequence: batch as single time step
        seq = z.unsqueeze(1)
        lstm_out, _ = self.lstm(seq)
        out = self.classifier(lstm_out.squeeze(1))
        return self.norm(out)

__all__ = ["HybridNatModel"]
