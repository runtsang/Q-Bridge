"""Hybrid model that marries classical convolution, quantum kernels,
autoencoding, and shot‑noise aware evaluation.

The class is intentionally lightweight: it only defines the network
architecture and a convenient evaluation routine.  Training and
hyper‑parameter tuning are left to the user, allowing integration with
common PyTorch pipelines."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from typing import Iterable, List, Sequence, Callable, Any

# --- FastEstimator utilities -----------------------------------------------
class FastEstimator:
    """Shot‑noise aware evaluator for deterministic PyTorch models."""
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor]],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Return a list of observable values for each parameter set.
        If ``shots`` is given, the outputs are Gaussian‑noised to emulate
        finite shot statistics."""
        self.model.eval()
        results: List[List[float]] = []
        rng = np.random.default_rng(seed)
        with torch.no_grad():
            for params in parameter_sets:
                inputs = torch.as_tensor(params, dtype=torch.float32).unsqueeze(0)
                outputs = self.model(inputs)
                row: List[float] = []
                for obs in observables:
                    value = obs(outputs)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)
                results.append(row)

        if shots is None:
            return results

        # Add shot noise
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

# --- Classical quanvolution filter -----------------------------------------
class ClassicalQuanvolutionFilter(nn.Module):
    """Simple 2×2 patch extraction followed by a linear projection."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ``x`` shape: (B, 1, 28, 28)
        features = self.conv(x)
        return features.view(x.size(0), -1)

# --- Quantum quanvolution kernel ------------------------------------------
try:
    import torchquantum as tq
except ImportError:
    tq = None  # pragma: no cover

class QuantumQuanvolutionFilter(nn.Module):
    """Two‑qubit quantum kernel applied to each 2×2 patch."""
    def __init__(self) -> None:
        super().__init__()
        if tq is None:
            raise RuntimeError("torchquantum is required for the quantum filter.")
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                data = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, data)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)

# --- Autoencoder ------------------------------------------------------------
class AutoencoderConfig:
    def __init__(self, input_dim: int, latent_dim: int = 32,
                 hidden_dims: Sequence[int] = (128, 64), dropout: float = 0.1):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout

class AutoencoderNet(nn.Module):
    def __init__(self, cfg: AutoencoderConfig) -> None:
        super().__init__()
        # Encoder
        enc_layers = []
        in_dim = cfg.input_dim
        for h in cfg.hidden_dims:
            enc_layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(cfg.dropout)]
            in_dim = h
        enc_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder
        dec_layers = []
        in_dim = cfg.latent_dim
        for h in reversed(cfg.hidden_dims):
            dec_layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(cfg.dropout)]
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(x))

# --- Hybrid model ----------------------------------------------------------
class HybridQuanvolutionAutoencoder(nn.Module):
    """
    Combines:

    * Classical or quantum quanvolution filter (configurable)
    * Fully‑connected autoencoder
    * Optional classification head

    The module is compatible with FastEstimator for shot‑noise simulation.
    """
    def __init__(self,
                 use_quantum_filter: bool = False,
                 autoencoder_cfg: AutoencoderConfig | None = None,
                 num_classes: int | None = None) -> None:
        super().__init__()
        self.use_quantum_filter = use_quantum_filter
        self.filter = QuantumQuanvolutionFilter() if use_quantum_filter else ClassicalQuanvolutionFilter()
        cfg = autoencoder_cfg or AutoencoderConfig(input_dim=4 * 14 * 14)
        self.autoencoder = AutoencoderNet(cfg)
        self.head = nn.Linear(cfg.latent_dim, num_classes) if num_classes else None

    def forward(self, x: torch.Tensor) -> Any:
        # Extract patch features
        feats = self.filter(x)
        # Autoencoder operates on flattened feature vector
        latent = self.autoencoder.encode(feats)
        recon = self.autoencoder.decode(latent)
        out = {"reconstruction": recon, "latent": latent}
        if self.head:
            out["logits"] = self.head(latent)
        return out

    def evaluate(self,
                 observables: Iterable[Callable[[Any], torch.Tensor | float]],
                 parameter_sets: Sequence[Sequence[float]],
                 *,
                 shots: int | None = None,
                 seed: int | None = None) -> List[List[float]]:
        """
        Evaluate a list of observable callables over the model.

        Observables receive the dictionary returned by ``forward``.
        """
        estimator = FastEstimator(self)
        return estimator.evaluate(observables, parameter_sets, shots=shots, seed=seed)

__all__ = ["HybridQuanvolutionAutoencoder", "AutoencoderNet", "AutoencoderConfig", "FastEstimator"]
