import torch
import numpy as np
from torch import nn
from typing import Iterable, List, Sequence, Callable

class AutoencoderConfig:
    def __init__(self, input_dim: int, latent_dim: int = 32, hidden_dims: Sequence[int] = (128, 64), dropout: float = 0.1):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout

class AutoencoderNet(nn.Module):
    def __init__(self, config: AutoencoderConfig):
        super().__init__()
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

def Autoencoder(input_dim: int, *, latent_dim: int = 32, hidden_dims: Sequence[int] = (128, 64), dropout: float = 0.1) -> AutoencoderNet:
    return AutoencoderNet(AutoencoderConfig(input_dim, latent_dim, hidden_dims, dropout))

class FastBaseEstimator:
    def __init__(self, model: nn.Module):
        self.model = model

    def evaluate(self, observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]], parameter_sets: Sequence[Sequence[float]]) -> List[List[float]]:
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = torch.as_tensor(params, dtype=torch.float32).unsqueeze(0)
                outputs = self.model(inputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().cpu())
                    else:
                        scalar = float(val)
                    row.append(scalar)
                results.append(row)
        return results

class FastEstimator(FastBaseEstimator):
    def evaluate(self, observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]], parameter_sets: Sequence[Sequence[float]], *, shots: int | None = None, seed: int | None = None) -> List[List[float]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

class HybridAttentionAutoencoder(nn.Module):
    def __init__(self, embed_dim: int, *, latent_dim: int = 32, hidden_dims: Sequence[int] = (128, 64), dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=1, batch_first=True)
        self.autoencoder = Autoencoder(embed_dim, latent_dim=latent_dim, hidden_dims=hidden_dims, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        flat = attn_out.reshape(attn_out.size(0), -1)
        latent = self.autoencoder.encode(flat)
        recon = self.autoencoder.decode(latent)
        return recon

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        flat = attn_out.reshape(attn_out.size(0), -1)
        return self.autoencoder.encode(flat)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.decode(z)

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, inputs: torch.Tensor) -> torch.Tensor:
        # Classical self‑attention using supplied parameters
        query = torch.as_tensor(inputs @ rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        key = torch.as_tensor(inputs @ entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        value = torch.as_tensor(inputs, dtype=torch.float32)
        scores = torch.softmax(query @ key.transpose(1, 2) / np.sqrt(self.embed_dim), dim=-1)
        attn_out = torch.bmm(scores, value)
        flat = attn_out.reshape(attn_out.size(0), -1)
        latent = self.autoencoder.encode(flat)
        recon = self.autoencoder.decode(latent)
        return recon

__all__ = ["HybridAttentionAutoencoder", "FastBaseEstimator", "FastEstimator"]
