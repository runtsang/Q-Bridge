from __future__ import annotations

import torch
from torch import nn
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple, Optional

# ----------------------------------------------------------------------
# Fraud‑style layer utilities (classical analogue)
# ----------------------------------------------------------------------
@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    weight = torch.tensor([[params.bs_theta, params.bs_phi],
                           [params.squeeze_r[0], params.squeeze_r[1]]], dtype=torch.float32)
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

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

# ----------------------------------------------------------------------
# EstimatorQNN (classical feed‑forward regressor)
# ----------------------------------------------------------------------
class EstimatorQNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)

# ----------------------------------------------------------------------
# HybridAutoencoder
# ----------------------------------------------------------------------
class HybridAutoencoder(nn.Module):
    """
    Hybrid autoencoder that chains fraud‑style linear layers for the encoder,
    optionally applies a quantum‑kernel mapping, and uses a classical EstimatorQNN
    as the decoder.  The design mirrors the classical Autoencoder, the quantum
    kernel construction, and the fraud‑detection style layer parametrisation
    from the reference pairs.
    """

    def __init__(
        self,
        encoder_params: Sequence[FraudLayerParameters],
        decoder_params: Sequence[FraudLayerParameters],
        latent_dim: int = 32,
        use_quantum_kernel: bool = False,
        reference_vectors: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        if not encoder_params:
            raise ValueError("At least one encoder layer parameter set must be supplied")
        self.encoder = build_fraud_detection_program(encoder_params[0], encoder_params[1:])
        self.latent_dim = latent_dim
        self.use_quantum_kernel = use_quantum_kernel
        self.reference_vectors = reference_vectors
        if use_quantum_kernel and reference_vectors is None:
            raise ValueError("reference_vectors must be supplied when use_quantum_kernel=True")

        # Projection from latent to 2‑dim space for the EstimatorQNN
        self.proj = nn.Linear(latent_dim, 2)
        self.decoder = EstimatorQNN()

    def _apply_quantum_kernel(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Compute a kernel similarity between each latent vector and the
        reference set.  The kernel is implemented as an RBF (classical)
        but is a placeholder for a true quantum kernel.  The resulting
        similarity vector is returned as the new latent representation.
        """
        gamma = 0.5
        diff = latent.unsqueeze(1) - self.reference_vectors.unsqueeze(0)
        dist2 = torch.sum(diff * diff, dim=-1)
        kernel = torch.exp(-gamma * dist2)
        # reduce to a single scalar per sample by averaging over references
        return kernel.mean(dim=-1, keepdim=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(inputs)
        if self.use_quantum_kernel:
            latent = self._apply_quantum_kernel(latent)
        projected = self.proj(latent)
        output = self.decoder(projected)
        return output

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the latent representation."""
        return self.encoder(inputs)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode a latent vector back to the output space."""
        projected = self.proj(latent)
        return self.decoder(projected)

__all__ = ["HybridAutoencoder", "FraudLayerParameters", "EstimatorQNN"]
