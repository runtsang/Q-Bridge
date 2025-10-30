import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Sequence, List, Callable


class QuanvolutionFilter(nn.Module):
    """Classical 2×2 patching followed by a 1‑to‑4 channel convolution."""
    def __init__(self, in_ch: int = 1, out_ch: int = 4, kernel: int = 2, stride: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, 1, 28, 28) → (B, 4, 14, 14) → (B, 4*14*14)
        return self.conv(x).view(x.size(0), -1)


class AutoencoderNet(nn.Module):
    """Simple fully‑connected autoencoder used to compress quanvolution features."""
    def __init__(self,
                 input_dim: int,
                 latent_dim: int = 32,
                 hidden_dims: Sequence[int] = (128, 64),
                 dropout: float = 0.1) -> None:
        super().__init__()
        enc_layers: List[nn.Module] = []
        dim = input_dim
        for h in hidden_dims:
            enc_layers += [nn.Linear(dim, h), nn.ReLU(), nn.Dropout(dropout)]
            dim = h
        enc_layers.append(nn.Linear(dim, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers: List[nn.Module] = []
        dim = latent_dim
        for h in reversed(hidden_dims):
            dec_layers += [nn.Linear(dim, h), nn.ReLU(), nn.Dropout(dropout)]
            dim = h
        dec_layers.append(nn.Linear(dim, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


class QuanvolutionHybrid(nn.Module):
    """Hybrid pipeline: quanvolution → autoencoder → linear classifier."""
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        # Feature size after quanvolution: 4 * 14 * 14
        self.autoencoder = AutoencoderNet(input_dim=4 * 14 * 14, latent_dim=32)
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.qfilter(x)
        compressed = self.autoencoder.encode(feats)
        logits = self.classifier(compressed)
        return F.log_softmax(logits, dim=-1)


class FastEstimator:
    """Convenience wrapper that evaluates a model on batched inputs."""
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(self,
                 observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
                 parameter_sets: Sequence[Sequence[float]],
                 *,
                 shots: int | None = None,
                 seed: int | None = None) -> List[List[float]]:
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inp = torch.as_tensor(params, dtype=torch.float32).unsqueeze(0)
                out = self.model(inp)
                row: List[float] = []
                for obs in observables:
                    val = obs(out)
                    if isinstance(val, torch.Tensor):
                        val = float(val.mean().cpu())
                    else:
                        val = float(val)
                    row.append(val)
                results.append(row)
        return results
