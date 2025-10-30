"""Hybrid fraud‑detection model combining classical layers, an auto‑encoder and a quantum feature map.

The module defines a single ``FraudDetectionHybrid`` class that
builds upon the original fraud‑detection layers, adds an auto‑encoder
encoder as a feature extractor and a PennyLane based quantum kernel.
It also exposes a lightweight ``FastEstimator`` evaluation routine that
adds configurable shot noise, mirroring the behaviour of the fast
estimator in the original QML seed.

The implementation is intentionally concise yet fully functional
for research‑grade experiments and can be dropped into any
PyTorch‑based training pipeline.
"""

from __future__ import annotations

import torch
from torch import nn
from dataclasses import dataclass
from typing import Iterable, Sequence, List, Callable, Tuple
import numpy as np

# ----------------------------------------------------------------------
# Auto‑encoder definitions (from the Autoencoder.py seed)
# ----------------------------------------------------------------------
@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

class AutoencoderNet(nn.Module):
    def __init__(self, config: AutoencoderConfig) -> None:
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

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(inputs))

def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
) -> AutoencoderNet:
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    return AutoencoderNet(config)


# ----------------------------------------------------------------------
# Fraud‑detection layer definition (from FraudDetection.py seed)
# ----------------------------------------------------------------------
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


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    weight = torch.tensor(
        [
            [params.bs_theta, params.bs_phi],
            [params.squeeze_r[0], params.squeeze_r[1]],
        ],
        dtype=torch.float32,
    )
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

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()


# ----------------------------------------------------------------------
# FastEstimator (from FastBaseEstimator.py seed)
# ----------------------------------------------------------------------
class FastEstimator:
    """Adds optional Gaussian shot noise to the deterministic estimator."""
    def __init__(self, model: nn.Module, shots: int | None = None, seed: int | None = None) -> None:
        self.model = model
        self.shots = shots
        self.seed = seed

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        obs = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = torch.tensor(params, dtype=torch.float32).unsqueeze(0)
                outputs = self.model(inputs)
                row: List[float] = []
                for o in obs:
                    val = o(outputs)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().cpu())
                    else:
                        scalar = float(val)
                    row.append(scalar)
                results.append(row)
        if self.shots is not None:
            rng = np.random.default_rng(self.seed)
            noisy = []
            for row in results:
                noisy.append([float(rng.normal(mean, max(1e-6, 1 / self.shots))) for mean in row])
            results = noisy
        return results


# ----------------------------------------------------------------------
# Hybrid fraud‑detection model
# ----------------------------------------------------------------------
class FraudDetectionHybrid(nn.Module):
    """Hybrid fraud‑detection model combining classical layers, an auto‑encoder and a quantum feature map."""
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        autoencoder_cfg: AutoencoderConfig | None = None,
        quantum_depth: int = 2,
        shot_noise: int | None = None,
    ) -> None:
        super().__init__()
        self.shot_noise = shot_noise

        # Classical feature extractor
        self.classical = nn.Sequential(
            _layer_from_params(input_params, clip=False),
            *(_layer_from_params(layer, clip=True) for layer in layers),
            nn.Linear(2, 1),  # final classifier head
        )

        # Optional auto‑encoder as a feature extractor
        if autoencoder_cfg is not None:
            self.autoencoder = Autoencoder(
                input_dim=autoencoder_cfg.input_dim,
                latent_dim=autoencoder_cfg.latent_dim,
                hidden_dims=autoencoder_cfg.hidden_dims,
                dropout=autoencoder_cfg.dropout,
            )
        else:
            self.autoencoder = None

        # Quantum feature map (PennyLane)
        try:
            import pennylane as qml
        except ImportError as exc:
            raise RuntimeError("PennyLane is required for the quantum feature map.") from exc

        dev = qml.device("default.qubit", wires=quantum_depth)

        @qml.qnode(dev, interface="torch")
        def quantum_circuit(x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
            # Simple RY–CNOT–RZ circuit per qubit
            for i in range(quantum_depth):
                qml.RY(x[i], wires=i)
                if i < quantum_depth - 1:
                    qml.CNOT(wires=[i, i + 1])
            for i, p in enumerate(params):
                qml.RZ(p, wires=i % quantum_depth)
            return qml.expval(qml.PauliZ(0))

        self.quantum_circuit = quantum_circuit
        self.quantum_params = nn.Parameter(torch.randn(quantum_depth))

        # Final classifier after quantum feature
        self.linear = nn.Linear(1, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Classical path
        x = self.classical(inputs)
        # Auto‑encoder encoding if present
        if self.autoencoder is not None:
            x = self.autoencoder.encode(x)
        # Quantum feature map
        qfeat = self.quantum_circuit(x.squeeze(-1), self.quantum_params)
        # Final linear layer
        out = self.linear(qfeat.unsqueeze(-1))
        return out

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Evaluate a list of scalar observables on batched parameter sets with optional shot noise."""
        estimator = FastEstimator(self, shots=self.shot_noise, seed=42)
        return estimator.evaluate(observables, parameter_sets)

    def train_model(
        self,
        data: torch.Tensor,
        *,
        epochs: int = 100,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        device: torch.device | None = None,
    ) -> List[float]:
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        dataset = torch.utils.data.TensorDataset(data)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.MSELoss()
        history: List[float] = []
        for _ in range(epochs):
            epoch_loss = 0.0
            for batch, in loader:
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                pred = self(batch)
                loss = loss_fn(pred, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch.size(0)
            epoch_loss /= len(dataset)
            history.append(epoch_loss)
        return history

__all__ = [
    "FraudLayerParameters",
    "FraudDetectionHybrid",
    "Autoencoder",
    "AutoencoderConfig",
    "AutoencoderNet",
]
