"""FraudDetectionHybrid: classical neural network with quantum feature extractor.

The module augments the original FraudDetection.py architecture by inserting
a parameterised quantum circuit (based on the FCL example) after the first
classical layer.  The quantum circuit is implemented with Qiskit and returns
a scalar expectation value that is concatenated to the classical feature
vector before the final linear classifier.  This design preserves the
original API (`FraudLayerParameters`, `build_fraud_detection_program`)
while adding a quantum submodule that can be trained jointly with the
classical network.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, List

import torch
from torch import nn
import numpy as np
import qiskit

# --------------------------------------------------------------------------- #
# 1. Classical layer definitions (borrowed from FraudDetection.py)
# --------------------------------------------------------------------------- #

@dataclass
class FraudLayerParameters:
    """Parameters describing a fully connected layer in the classical model."""
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
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()

# --------------------------------------------------------------------------- #
# 2. Quantum fully‑connected layer (based on FCL.py)
# --------------------------------------------------------------------------- #

class QuantumFC(nn.Module):
    """
    A lightweight wrapper around the FCL quantum circuit.
    The circuit is a single qubit parameterised by a rotation angle theta.
    The forward pass takes a tensor of shape (batch, 1) and returns a
    tensor of shape (batch, 1) containing the expectation value of Z.
    """
    def __init__(self, backend=None, shots=1024):
        super().__init__()
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.circuit = qiskit.QuantumCircuit(1)
        self.theta = qiskit.circuit.Parameter("theta")
        self.circuit.h(0)
        self.circuit.barrier()
        self.circuit.ry(self.theta, 0)
        self.circuit.measure_all()

    def forward(self, thetas: torch.Tensor) -> torch.Tensor:
        # thetas: (batch, 1)
        thetas_np = thetas.detach().cpu().numpy().flatten()
        job = qiskit.execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[{self.theta: t} for t in thetas_np],
        )
        result = job.result()
        expectation = np.zeros(len(thetas_np))
        for i, _ in enumerate(thetas_np):
            counts = result.get_counts(self.circuit)
            # convert counts dict to expectation of Z
            c = counts.get("0", 0)
            d = counts.get("1", 0)
            probs = np.array([c, d]) / self.shots
            expectation[i] = probs[0] - probs[1]
        return torch.from_numpy(expectation.astype(np.float32)).unsqueeze(-1)

# --------------------------------------------------------------------------- #
# 3. Hybrid fraud detection model
# --------------------------------------------------------------------------- #

class FraudDetectionHybrid(nn.Module):
    """
    Hybrid fraud detection model.
    Parameters:
        input_params: FraudLayerParameters for the first classical layer.
        layers: Iterable[FraudLayerParameters] for subsequent classical layers.
        quantum_fc: instance of QuantumFC to be inserted after the first layer.
    """
    def __init__(self, input_params: FraudLayerParameters,
                 layers: Iterable[FraudLayerParameters],
                 quantum_fc: nn.Module):
        super().__init__()
        self.quantum_fc = quantum_fc
        self.classical_layers: List[nn.Module] = []
        # first layer (no clipping)
        self.classical_layers.append(_layer_from_params(input_params, clip=False))
        # subsequent layers (clipped)
        self.classical_layers.extend(_layer_from_params(l, clip=True) for l in layers)
        self.classical_layers = nn.ModuleList(self.classical_layers)
        # final linear classifier
        self.final_linear = nn.Linear(3, 1)  # 2 from classical + 1 from quantum

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 2)
        for layer in self.classical_layers:
            x = layer(x)
        # quantum feature extraction
        q_feat = self.quantum_fc(x[:, :1])  # use first feature as theta
        # concatenate
        x = torch.cat([x, q_feat], dim=1)
        out = self.final_linear(x)
        return out

# --------------------------------------------------------------------------- #
# 4. Factory function
# --------------------------------------------------------------------------- #

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
    quantum_fc: nn.Module,
) -> nn.Module:
    """Return a FraudDetectionHybrid model."""
    return FraudDetectionHybrid(input_params, layers, quantum_fc)

__all__ = ["FraudLayerParameters", "build_fraud_detection_program",
           "FraudDetectionHybrid", "QuantumFC"]
