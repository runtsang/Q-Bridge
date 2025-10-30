"""HybridSamplerQNN: Classical‑Quantum Fusion

This module defines a PyTorch neural network that first runs a quantum
sampler (provided by the QML counterpart) and then processes the
sampler's probability distribution through a stack of parameterised
classical layers inspired by the fraud‑detection seed.  The layers
include optional clipping and scaling, mirroring the photonic
implementation, while the quantum part supplies a learnable
probability vector that is fed into the classical stack.

The class can be instantiated with any Qiskit SamplerQNN object
and a sequence of FraudLayerParameters, enabling end‑to‑end
training with gradient back‑propagation through the quantum
circuit via qiskit‑machine‑learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Iterable, Sequence

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
            out = self.activation(self.linear(inputs))
            out = out * self.scale + self.shift
            return out

    return Layer()

def build_hybrid_network(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

class HybridSamplerQNN(nn.Module):
    """Classical‑Quantum hybrid sampler.

    Parameters
    ----------
    quantum_sampler : qiskit_machine_learning.neural_networks.SamplerQNN
        The quantum sampler that produces a probability vector.
    fraud_params : Iterable[FraudLayerParameters]
        Parameter tuples that build the classical stack.
    device : torch.device, optional
        Target device for tensors.
    """
    def __init__(
        self,
        quantum_sampler,
        fraud_params: Iterable[FraudLayerParameters],
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.quantum_sampler = quantum_sampler
        params_list = list(fraud_params)
        self.classical_net = build_hybrid_network(
            params_list[0], params_list[1:]
        )
        self.device = device or torch.device("cpu")
        self.to(self.device)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        inputs : torch.Tensor
            Shape (..., 2).  These are interpreted as the input
            parameters for the quantum sampler and are passed directly
            to its ``sample`` method.  The sampler returns a 4‑dim
            probability vector which is then flattened and fed into
            the classical network.

        Returns
        -------
        torch.Tensor
            Output of the classical stack (shape (..., 1)).
        """
        input_np = inputs.detach().cpu().numpy()
        probs = self.quantum_sampler.sample(input_np)
        probs_t = torch.from_numpy(probs).float().to(self.device)
        probs_t = probs_t.reshape(*probs_t.shape[:-1], 2)
        return self.classical_net(probs_t)

__all__ = ["FraudLayerParameters", "HybridSamplerQNN"]
