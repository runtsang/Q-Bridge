import torch
from torch import nn

class ConvAutoencoderHybrid(nn.Module):
    """
    Classical implementation of a convolutional autoencoder.
    The first layer is a 2‑D convolution that acts as a feature extractor.
    The output of the convolution is flattened and fed into a dense autoencoder.
    """
    def __init__(self,
                 conv_kernel_size: int = 2,
                 conv_threshold: float = 0.0,
                 latent_dim: int = 32,
                 hidden_dims: tuple[int, int] = (128, 64),
                 dropout: float = 0.1):
        super().__init__()
        self.conv_kernel_size = conv_kernel_size
        self.conv_threshold = conv_threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=conv_kernel_size, bias=True)

        self.conv_out_size = 28 - conv_kernel_size + 1
        flatten_dim = self.conv_out_size ** 2

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], flatten_dim),
            nn.Unflatten(1, (1, self.conv_out_size, self.conv_out_size))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_out = torch.sigmoid(self.conv(x) - self.conv_threshold)
        latent = self.encoder(conv_out)
        recon = self.decoder(latent)
        return recon

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        conv_out = torch.sigmoid(self.conv(x) - self.conv_threshold)
        return self.encoder(conv_out)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def training_step(self, batch: torch.Tensor) -> torch.Tensor:
        loss_fn = nn.MSELoss()
        recon = self.forward(batch)
        loss = loss_fn(recon, batch)
        return loss
