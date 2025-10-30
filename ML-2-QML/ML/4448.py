import torch
from torch import nn
import numpy as np

class AutoencoderNet(nn.Module):
    def __init__(self, input_dim=128, latent_dim=32, hidden_dims=(128, 64), dropout=0.1):
        super().__init__()
        encoder_layers = []
        in_dim = input_dim
        for hidden in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if dropout > 0.0:
                encoder_layers.append(nn.Dropout(dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = latent_dim
        for hidden in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if dropout > 0.0:
                decoder_layers.append(nn.Dropout(dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

class ClassicalSelfAttention:
    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        query = torch.as_tensor(inputs @ rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        key = torch.as_tensor(inputs @ entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        value = torch.as_tensor(inputs, dtype=torch.float32)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ value).numpy()

class EstimatorQNNGen128(nn.Module):
    def __init__(self, input_dim=128, latent_dim=32, hidden_dims=(128, 64), dropout=0.1):
        super().__init__()
        self.autoencoder = AutoencoderNet(input_dim, latent_dim, hidden_dims, dropout)
        self.attention = ClassicalSelfAttention(embed_dim=latent_dim)
        self.att_rotation = nn.Parameter(torch.randn(latent_dim * 3))
        self.att_entangle = nn.Parameter(torch.randn(latent_dim - 1))
        self.regressor = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.autoencoder.encode(x)
        latent_np = latent.detach().cpu().numpy()
        att_out_np = self.attention.run(
            self.att_rotation.detach().cpu().numpy(),
            self.att_entangle.detach().cpu().numpy(),
            latent_np
        )
        att_out = torch.as_tensor(att_out_np, device=x.device, dtype=x.dtype)
        out = self.regressor(att_out)
        return out

def EstimatorQNN():
    return EstimatorQNNGen128()

__all__ = ["EstimatorQNNGen128", "EstimatorQNN"]
