from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Callable, List, Tuple

import torch
from torch import nn
import numpy as np

# --------------------------------------------------------------------------- #
# 1. Parameter containers
# --------------------------------------------------------------------------- #

@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1
    fraud_layers: Sequence[FraudLayerParameters] | None = None

# --------------------------------------------------------------------------- #
# 2. Utility helpers
# --------------------------------------------------------------------------- #

def _to_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    t = torch.as_tensor(data, dtype=torch.float32)
    if t.ndim == 1:
        t = t.unsqueeze(0)
    return t

def _fraud_layer(params: FraudLayerParameters, clip: bool = True) -> nn.Module:
    weight = torch.tensor(
        [[params.bs_theta, params.bs_phi],
         [params.squeeze_r[0], params.squeeze_r[1]]],
        dtype=torch.float32)
    bias = torch.tensor(params.phases, dtype=torch.float32)
    if clip:
        weight = weight.clamp(-5.0, 5.0)
        bias = bias.clamp(-5.0, 5.0)
    linear = nn.Linear(2, 2)
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)
    activation = nn.Tanh()
    scale = torch.tensor(params.displacement_r, dtype=torch.float32)
    shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            y = self.activation(self.linear(x))
            y = y * self.scale + self.shift
            return y

    return Layer()

# --------------------------------------------------------------------------- #
# 3. Hybrid auto‑encoder
# --------------------------------------------------------------------------- #

class HybridAutoencoder(nn.Module):
    """A classical auto‑encoder that optionally incorporates fraud‑detection style layers
    and exposes a FastBaseEstimator‑like API for batched evaluation and noisy observation
    generation."""
    def __init__(self, cfg: AutoencoderConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Encoder
        encoder_modules: List[nn.Module] = []
        if cfg.fraud_layers:
            # first fraud layer is used as the input layer
            encoder_modules.append(_fraud_layer(cfg.fraud_layers[0], clip=False))
            for params in cfg.fraud_layers[1:]:
                encoder_modules.append(_fraud_layer(params, clip=True))
        else:
            in_dim = cfg.input_dim
            for hidden in cfg.hidden_dims:
                encoder_modules.append(nn.Linear(in_dim, hidden))
                encoder_modules.append(nn.ReLU())
                if cfg.dropout > 0.0:
                    encoder_modules.append(nn.Dropout(cfg.dropout))
                in_dim = hidden
        encoder_modules.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*encoder_modules)

        # Decoder (mirror of encoder but with optional fraud layers)
        decoder_modules: List[nn.Module] = []
        if cfg.fraud_layers:
            # use the same fraud layers in reverse order
            for params in reversed(cfg.fraud_layers):
                decoder_modules.append(_fraud_layer(params, clip=True))
            decoder_modules.append(nn.Linear(cfg.latent_dim, cfg.input_dim))
        else:
            in_dim = cfg.latent_dim
            for hidden in reversed(cfg.hidden_dims):
                decoder_modules.append(nn.Linear(in_dim, hidden))
                decoder_modules.append(nn.ReLU())
                if cfg.dropout > 0.0:
                    decoder_modules.append(nn.Dropout(cfg.dropout))
                in_dim = hidden
            decoder_modules.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*decoder_modules)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(x))

    # --------------------------------------------------------------------- #
    # Estimator interface
    # --------------------------------------------------------------------- #
    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        params: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Evaluate a list of scalar observables for each parameter set.
        If *shots* is provided, a Gaussian noise term is added to emulate finite‑shot statistics."""
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []

        self.eval()
        with torch.no_grad():
            for p_set in params:
                x = _to_tensor(p_set)
                out = self(x)
                row: List[float] = []
                for obs in observables:
                    val = obs(out)
                    if isinstance(val, torch.Tensor):
                        val = float(val.mean().cpu())
                    else:
                        val = float(val)
                    row.append(val)
                results.append(row)

        if shots is not None:
            rng = np.random.default_rng(seed)
            noisy: List[List[float]] = []
            for row in results:
                noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
                noisy.append(noisy_row)
            return noisy

        return results

# --------------------------------------------------------------------------- #
# 4. Training helper
# --------------------------------------------------------------------------- #

def train_hybrid_autoencoder(
    model: HybridAutoencoder,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> List[float]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = torch.utils.data.TensorDataset(_to_tensor(data))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: List[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            opt.zero_grad(set_to_none=True)
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

__all__ = ["HybridAutoencoder", "AutoencoderConfig", "FraudLayerParameters", "train_hybrid_autoencoder"]
