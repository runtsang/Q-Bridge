"""Hybrid model combining classical quanvolution filtering, QCNN-inspired linear layers,
autoencoder compression, and optional LSTM for sequence modeling."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionHybrid(nn.Module):
    """Hybrid classical model mirroring the quantum‑enhanced counterpart."""
    def __init__(
        self,
        num_classes: int = 10,
        use_lstm: bool = False,
        lstm_hidden: int = 64,
        seq_len: int = 1,
        autoencoder_latent: int = 32,
        autoencoder_hidden: tuple[int, int] = (128, 64),
    ) -> None:
        super().__init__()
        # 1. Classical quanvolution filter: 2×2 patches → 4 channels
        self.qfilter = nn.Conv2d(1, 4, kernel_size=2, stride=2)
        # 2. QCNN‑style linear stack
        self.qcnn = nn.Sequential(
            nn.Linear(4 * 14 * 14, 256), nn.Tanh(),
            nn.Linear(256, 128), nn.Tanh(),
            nn.Linear(128, 64), nn.Tanh(),
        )
        # 3. Autoencoder for dimensionality reduction
        self.autoencoder = self._make_autoencoder(autoencoder_latent, autoencoder_hidden)
        # 4. Optional LSTM for sequence modelling
        self.use_lstm = use_lstm
        if use_lstm:
            self.lstm = nn.LSTM(input_size=autoencoder_latent,
                                hidden_size=lstm_hidden,
                                batch_first=True)
        # 5. Classification head
        final_dim = lstm_hidden if use_lstm else autoencoder_latent
        self.classifier = nn.Linear(final_dim, num_classes)

    def _make_autoencoder(self, latent_dim: int, hidden_dims: tuple[int, int]) -> nn.ModuleDict:
        encoder_layers = []
        in_dim = 4 * 14 * 14  # output of qfilter before flatten
        for h in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, h))
            encoder_layers.append(nn.ReLU())
            in_dim = h
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = latent_dim
        for h in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, h))
            decoder_layers.append(nn.ReLU())
            in_dim = h
        decoder_layers.append(nn.Linear(in_dim, 4 * 14 * 14))
        decoder = nn.Sequential(*decoder_layers)

        return nn.ModuleDict({"encoder": encoder, "decoder": decoder})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, 1, 28, 28) or (batch, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Log‑softmax logits of shape (batch, num_classes).
        """
        if x.dim() == 5:
            batch, seq_len, c, h, w = x.shape
            x = x.view(batch * seq_len, c, h, w)
        else:
            seq_len = 1
        # 1. Quanvolution
        feat = self.qfilter(x)  # (batch*seq_len, 4, 14, 14)
        feat = feat.view(x.shape[0], -1)  # flatten
        # 2. QCNN block
        feat = self.qcnn(feat)
        # 3. Autoencoder encoding
        latent = self.autoencoder["encoder"](feat)
        # 4. Sequence modelling
        if seq_len > 1:
            latent = latent.view(batch, seq_len, -1)
            latent, _ = self.lstm(latent)
            latent = latent[:, -1, :]
        # 5. Classification
        logits = self.classifier(latent)
        return F.log_softmax(logits, dim=-1)

    def decode_autoencoder(self, latent: torch.Tensor) -> torch.Tensor:
        """Reconstruct features from latent representation."""
        return self.autoencoder["decoder"](latent)
