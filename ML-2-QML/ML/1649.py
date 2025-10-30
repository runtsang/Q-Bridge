import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionGen224(nn.Module):
    """
    A convolutional auto‑encoder that first extracts 2×2 patches via a
    quanvolution‑style convolution, then compresses the representation
    to 224 latent dimensions and finally reconstructs the original
    28×28 image.  The encoder mirrors the original filter but adds a
    second convolution and a fully‑connected bottleneck, while the
    decoder mirrors the encoder via transposed convolutions.
    """
    def __init__(self) -> None:
        super().__init__()
        # Encoder
        self.conv1 = nn.Conv2d(1, 4, kernel_size=2, stride=2)   # 28x28 -> 14x14
        self.conv2 = nn.Conv2d(4, 8, kernel_size=2, stride=2)   # 14x14 -> 7x7
        self.flatten = nn.Flatten()
        self.enc_fc = nn.Linear(8 * 7 * 7, 224)

        # Decoder
        self.dec_fc = nn.Linear(224, 8 * 7 * 7)
        self.unflatten = nn.Unflatten(1, (8, 7, 7))
        self.deconv1 = nn.ConvTranspose2d(8, 4, kernel_size=2, stride=2)  # 7x7 -> 14x14
        self.deconv2 = nn.ConvTranspose2d(4, 1, kernel_size=2, stride=2)  # 14x14 -> 28x28

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image batch of shape (batch, 1, 28, 28).
        Returns:
            Reconstructed images of shape (batch, 1, 28, 28).
        """
        # Encode
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.flatten(x)
        latent = F.relu(self.enc_fc(x))

        # Decode
        x = F.relu(self.dec_fc(latent))
        x = self.unflatten(x)
        x = F.relu(self.deconv1(x))
        x = torch.sigmoid(self.deconv2(x))
        return x
