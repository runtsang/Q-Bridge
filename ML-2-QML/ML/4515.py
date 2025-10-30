from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    t = torch.as_tensor(values, dtype=torch.float32)
    if t.ndim == 1:
        t = t.unsqueeze(0)
    return t

class SharedClassName(nn.Module):
    """Hybrid estimator that unifies classical neural‑network evaluation with quantum‑inspired utilities."""
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.model.eval()

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                row: List[float] = []
                for observable in observables:
                    val = observable(outputs)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().cpu())
                    else:
                        scalar = float(val)
                    row.append(scalar)
                results.append(row)
        if shots is None:
            return results
        rng = np.random.default_rng(seed)
        noisy = []
        for row in results:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

    # Fraud detection utilities -----------------------------------------
    @staticmethod
    def fraud_layer_from_params(params: "FraudLayerParameters", clip: bool = False) -> nn.Module:
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

    @staticmethod
    def build_fraud_detection_program(
        input_params: "FraudLayerParameters",
        layers: Iterable["FraudLayerParameters"],
    ) -> nn.Sequential:
        modules = [SharedClassName.fraud_layer_from_params(input_params, clip=False)]
        modules.extend(SharedClassName.fraud_layer_from_params(layer, clip=True) for layer in layers)
        modules.append(nn.Linear(2, 1))
        return nn.Sequential(*modules)

    # Convolution filter --------------------------------------------------
    @staticmethod
    def conv(kernel_size: int = 2, threshold: float = 0.0) -> nn.Module:
        class ConvFilter(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.kernel_size = kernel_size
                self.threshold = threshold
                self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

            def run(self, data) -> float:
                tensor = torch.as_tensor(data, dtype=torch.float32)
                tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
                logits = self.conv(tensor)
                activations = torch.sigmoid(logits - self.threshold)
                return activations.mean().item()

        return ConvFilter()

    # Autoencoder ---------------------------------------------------------
    @staticmethod
    def autoencoder(
        input_dim: int,
        *,
        latent_dim: int = 32,
        hidden_dims: Tuple[int, int] = (128, 64),
        dropout: float = 0.1,
    ) -> nn.Module:
        class AutoencoderNet(nn.Module):
            def __init__(self) -> None:
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

            def encode(self, inputs: torch.Tensor) -> torch.Tensor:
                return self.encoder(inputs)

            def decode(self, latents: torch.Tensor) -> torch.Tensor:
                return self.decoder(latents)

            def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
                return self.decode(self.encode(inputs))

        return AutoencoderNet()

    @staticmethod
    def train_autoencoder(
        model: nn.Module,
        data: torch.Tensor,
        *,
        epochs: int = 100,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        device: torch.device | None = None,
    ) -> list[float]:
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        dataset = TensorDataset(_as_tensor(data))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.MSELoss()
        history: list[float] = []

        for _ in range(epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                reconstruction = model(batch)
                loss = loss_fn(reconstruction, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch.size(0)
            epoch_loss /= len(dataset)
            history.append(epoch_loss)
        return history

# Helper function for tensor conversion
def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

# Dataclass for fraud detection parameters
from dataclasses import dataclass

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
