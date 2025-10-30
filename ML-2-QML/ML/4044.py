import torch
from torch import nn
import numpy as np

class HybridFCL(nn.Module):
    """
    Hybrid classical fully‑connected layer that chains a 2‑D convolution,
    a single‑feature linear layer and a lightweight auto‑encoder.
    """
    def __init__(self,
                 conv_kernel: int = 2,
                 conv_threshold: float = 0.0,
                 fc_n_features: int = 1,
                 latent_dim: int = 32,
                 hidden_dims: tuple[int, int] = (128, 64),
                 dropout: float = 0.1):
        super().__init__()
        # Convolutional front‑end
        self.conv = nn.Conv2d(1, 1, kernel_size=conv_kernel, bias=True)
        self.conv_threshold = conv_threshold
        # Fully‑connected core
        self.fc = nn.Linear(fc_n_features, 1)
        # Auto‑encoder body
        enc_layers = []
        in_dim = hidden_dims[0]
        for hidden in hidden_dims:
            enc_layers.append(nn.Linear(in_dim, hidden))
            enc_layers.append(nn.ReLU())
            if dropout > 0.0:
                enc_layers.append(nn.Dropout(dropout))
            in_dim = hidden
        enc_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)
        dec_layers = []
        in_dim = latent_dim
        for hidden in reversed(hidden_dims):
            dec_layers.append(nn.Linear(in_dim, hidden))
            dec_layers.append(nn.ReLU())
            if dropout > 0.0:
                dec_layers.append(nn.Dropout(dropout))
            in_dim = hidden
        dec_layers.append(nn.Linear(in_dim, 1))
        self.decoder = nn.Sequential(*dec_layers)

    def conv_forward(self, x: np.ndarray) -> torch.Tensor:
        x_t = torch.as_tensor(x, dtype=torch.float32).view(1, 1, self.conv.kernel_size, self.conv.kernel_size)
        logits = self.conv(x_t)
        act = torch.sigmoid(logits - self.conv_threshold)
        return act.mean()

    def fc_forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.fc(x.view(-1, 1))).mean()

    def autoencoder_forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon.squeeze()

    def forward(self, data: np.ndarray):
        """
        Execute the full hybrid pipeline.

        Parameters
        ----------
        data : np.ndarray
            2‑D array of shape (conv_kernel, conv_kernel).

        Returns
        -------
        Tuple[float, float, float]
            (conv_out, fc_out, recon_out)
        """
        conv_out = self.conv_forward(data)
        fc_out = self.fc_forward(conv_out.unsqueeze(0))
        recon = self.autoencoder_forward(fc_out)
        return conv_out.item(), fc_out.item(), recon.item()
