"""
Hybrid classical model combining a convolution filter and an autoencoder.
Designed to be a drop‑in replacement for the legacy Conv.py functionality.
"""

from __future__ import annotations

import torch
from torch import nn

# Import the reusable components defined in the original seeds
from Conv import Conv          # Classical convolution filter
from Autoencoder import Autoencoder, AutoencoderConfig
from Autoencoder import train_autoencoder  # training helper


class HybridConvAE(nn.Module):
    """Hybrid convolution + autoencoder.

    The module first applies a single‑channel convolutional filter
    and then feeds the resulting scalar feature into a classical
    fully‑connected autoencoder.  The API mirrors the original
    Conv.py and Autoencoder.py modules so that existing training
    scripts can import and use :class:`HybridConvAE` without
    modification.
    """

    def __init__(
        self,
        conv_kernel: int = 2,
        conv_threshold: float = 0.0,
        ae_config: AutoencoderConfig | None = None,
    ) -> None:
        super().__init__()
        self.conv = Conv(kernel_size=conv_kernel, threshold=conv_threshold)

        # Default to a 1‑dimensional autoencoder (scalar input)
        default_cfg = AutoencoderConfig(
            input_dim=1,
            latent_dim=32,
            hidden_dims=(128, 64),
            dropout=0.1,
        )
        cfg = ae_config or default_cfg
        self.autoencoder = Autoencoder(
            input_dim=cfg.input_dim,
            latent_dim=cfg.latent_dim,
            hidden_dims=cfg.hidden_dims,
            dropout=cfg.dropout,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        inputs : torch.Tensor
            2‑D tensor of shape (kernel_size, kernel_size).

        Returns
        -------
        torch.Tensor
            Reconstructed scalar output from the autoencoder.
        """
        # The original ConvFilter returns a scalar float
        conv_val = self.conv.run(inputs)
        conv_tensor = torch.as_tensor(conv_val, dtype=torch.float32).unsqueeze(0)
        return self.autoencoder(conv_tensor)

    def train_epoch(
        self,
        data: torch.Tensor,
        *,
        batch_size: int = 64,
        lr: float = 1e-3,
        epochs: int = 10,
        device: torch.device | None = None,
    ) -> list[float]:
        """
        Convenience training loop that mirrors the legacy Autoencoder
        training routine but operates on the hybrid forward method.

        Returns a list of epoch‑wise MSE losses.
        """
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        dataset = torch.utils.data.TensorDataset(data)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        history: list[float] = []

        for _ in range(epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                recon = self(batch)
                loss = loss_fn(recon, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch.size(0)
            epoch_loss /= len(loader.dataset)
            history.append(epoch_loss)
        return history


__all__ = ["HybridConvAE"]
