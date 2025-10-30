import torch
import numpy as np
from torch import nn

class AutoencoderConfig:
    """Configuration for the PyTorch auto‑encoder."""
    def __init__(self, input_dim, latent_dim=32, hidden_dims=(128, 64), dropout=0.1):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout

class AutoencoderNet(nn.Module):
    """Standard MLP auto‑encoder."""
    def __init__(self, config: AutoencoderConfig):
        super().__init__()
        # Encoder
        encoder_layers = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

class HybridAutoencoderLayer(nn.Module):
    """
    Classical hybrid layer that combines a PyTorch auto‑encoder with a
    quantum‑style fully‑connected output.  The final layer applies a
    tanh activation to a linear projection of the latent vector,
    mimicking the expectation value of a simple quantum circuit.
    """
    def __init__(self,
                 input_dim: int,
                 latent_dim: int = 32,
                 hidden_dims: tuple[int, int] = (128, 64),
                 dropout: float = 0.1):
        super().__init__()
        config = AutoencoderConfig(input_dim, latent_dim, hidden_dims, dropout)
        self.autoencoder = AutoencoderNet(config)
        self.fc = nn.Linear(latent_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the quantum‑style output for the input batch."""
        latent = self.autoencoder.encode(x)
        return torch.tanh(self.fc(latent))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the latent representation."""
        return self.autoencoder.encode(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct from latent vector."""
        return self.autoencoder.decode(z)

    def run(self, thetas: np.ndarray, batch: torch.Tensor) -> np.ndarray:
        """
        Compute a classical expectation analogous to the quantum
        fully‑connected layer.  `thetas` must be a 1‑D array with
        length equal to the number of weights in `self.fc`.  The
        method applies the linear transform to the latent vector of
        `batch`, then returns the mean tanh activation.
        """
        if len(thetas)!= self.fc.weight.numel():
            raise ValueError("Theta length does not match layer weights.")
        weight = torch.from_numpy(thetas).view(self.fc.weight.size())
        bias = self.fc.bias
        latent = self.autoencoder.encode(batch.to(weight.device))
        transformed = torch.matmul(latent, weight.t()) + bias
        expectation = torch.tanh(transformed).mean(dim=0)
        return expectation.detach().cpu().numpy()
