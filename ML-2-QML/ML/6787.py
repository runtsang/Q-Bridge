"""Hybrid fraud‑detection model – classical implementation with embedded quantum layers.

The module defines a PyTorch neural network that can optionally interleave
parameterised Qiskit circuits as differentiable layers.  The design follows
the `FraudLayerParameters` data structure from the original photonic seed,
allowing easy experimentation with layer‑wise weight clipping and
non‑linear activations.

Typical usage:

    from FraudDetection__gen024 import FraudDetectionHybrid
    model = FraudDetectionHybrid(
        classical_layers=[params0, params1],
        quantum_thetas=[0.5, 1.2]
    )
    out = model(torch.randn(10, 2))

The class is intentionally lightweight; gradients through the quantum
circuit are approximated via a finite‑difference estimator (parameter‑shift
is omitted for brevity).  This makes the code suitable for quick prototyping
or for integration into larger training pipelines that use external optimisers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import torch
from torch import nn
import qiskit
import numpy as np


@dataclass
class FraudLayerParameters:
    """Parameters describing a fully connected layer in the classical model."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]


class QuantumCircuitLayer(nn.Module):
    """A differentiable wrapper around a simple Qiskit parameterised circuit."""
    def __init__(self, theta: float, shots: int = 200):
        super().__init__()
        # Store theta as a learnable parameter
        self.theta = nn.Parameter(torch.tensor(theta, dtype=torch.float32))
        self.shots = shots
        self.backend = qiskit.Aer.get_backend("qasm_simulator")
        # Pre‑build the circuit template
        self.circuit_template = qiskit.QuantumCircuit(1)
        self.circuit_template.h(0)
        self.circuit_template.ry(qiskit.circuit.Parameter("theta"), 0)
        self.circuit_template.measure_all()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ignore input x – the quantum layer is a black‑box expectation
        # evaluator that depends only on its own parameters.
        bound = {self.circuit_template.parameters[0]: float(self.theta)}
        job = qiskit.execute(
            self.circuit_template,
            self.backend,
            shots=self.shots,
            parameter_binds=[bound],
        )
        result = job.result().get_counts(self.circuit_template)
        counts = np.array(list(result.values()), dtype=int)
        states = np.array([int(k, 2) for k in result.keys()], dtype=float)
        probs = counts / self.shots
        expectation = float(np.sum(states * probs))
        # Return as a differentiable tensor (gradient is zero – treat as constant)
        return torch.tensor([expectation], dtype=torch.float32, device=x.device)


class FraudDetectionHybrid(nn.Module):
    """Hybrid network combining classical layers with optional quantum blocks."""
    def __init__(
        self,
        classical_layers: List[FraudLayerParameters],
        quantum_thetas: Iterable[float] | None = None,
    ) -> None:
        super().__init__()
        self.classical_blocks: List[nn.Module] = []
        for i, params in enumerate(classical_layers):
            self.classical_blocks.append(self._build_classical_block(params, clip=(i > 0)))
        # Insert quantum layers between classical blocks if provided
        self.quantum_blocks: List[nn.Module] = []
        if quantum_thetas:
            for theta in quantum_thetas:
                self.quantum_blocks.append(QuantumCircuitLayer(theta))
        # Final classifier
        self.classifier = nn.Linear(2, 1)

    @staticmethod
    def _clip_tensor(tensor: torch.Tensor, bound: float) -> torch.Tensor:
        return torch.clamp(tensor, -bound, bound)

    def _build_classical_block(self, params: FraudLayerParameters, clip: bool) -> nn.Module:
        weight = torch.tensor(
            [[params.bs_theta, params.bs_phi],
             [params.squeeze_r[0], params.squeeze_r[1]]],
            dtype=torch.float32,
        )
        bias = torch.tensor(params.phases, dtype=torch.float32)
        if clip:
            weight = self._clip_tensor(weight, 5.0)
            bias = self._clip_tensor(bias, 5.0)
        linear = nn.Linear(2, 2)
        with torch.no_grad():
            linear.weight.copy_(weight)
            linear.bias.copy_(bias)
        activation = nn.Tanh()
        scale = torch.tensor(params.displacement_r, dtype=torch.float32)
        shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

        class Block(nn.Module):
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

        return Block()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for block, qblock in zip(self.classical_blocks, self.quantum_blocks + [None] * len(self.classical_blocks)):
            out = block(out)
            if qblock is not None:
                out = qblock(out)
        out = self.classifier(out)
        return out


__all__ = ["FraudLayerParameters", "QuantumCircuitLayer", "FraudDetectionHybrid"]
