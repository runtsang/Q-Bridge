"""FraudDetectionHybrid: quantum-classical hybrid model implemented with PennyLane.

The quantum part is a 2‑qubit variational circuit that mirrors the photonic
operations from the original Strawberry Fields code: beam‑splitters, phase
shifts, squeezers, displacements and Kerr non‑linearities are mapped to
qubit gates (RY, RZ, CX, etc.).  The circuit outputs expectation values of
Pauli‑Z for each mode, which are concatenated with the classical feature
vector and passed through a linear classifier.  This design keeps the
original `FraudLayerParameters` interface while providing a fully
quantum‑aware feature extractor that can be trained end‑to‑end.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, List

import torch
from torch import nn
import pennylane as qml

# --------------------------------------------------------------------------- #
# 1. Classical layer definitions (borrowed from FraudDetection.py)
# --------------------------------------------------------------------------- #

@dataclass
class FraudLayerParameters:
    """Parameters describing a single photonic layer."""
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
# 2. Variational quantum circuit (based on Strawberry Fields design)
# --------------------------------------------------------------------------- #

def _quantum_layer(params: FraudLayerParameters, device: qml.Device):
    @qml.qnode(device, interface="torch")
    def circuit(inputs: torch.Tensor):
        # inputs: (batch, 2) classical features
        # Map classical inputs to rotation angles
        theta0 = inputs[:, 0]
        theta1 = inputs[:, 1]
        # Beam splitter like entanglement
        qml.RY(params.bs_theta, wires=0)
        qml.RY(params.bs_phi, wires=1)
        qml.CX(wires=[0, 1])
        # Phase shifts
        qml.RZ(params.phases[0], wires=0)
        qml.RZ(params.phases[1], wires=1)
        # Squeezing mapped to RY
        qml.RY(params.squeeze_r[0], wires=0)
        qml.RY(params.squeeze_r[1], wires=1)
        # Displacement mapped to RZ
        qml.RZ(params.displacement_r[0], wires=0)
        qml.RZ(params.displacement_r[1], wires=1)
        # Kerr mapped to small RZ
        qml.RZ(params.kerr[0], wires=0)
        qml.RZ(params.kerr[1], wires=1)
        # Expectation values of Pauli‑Z on both qubits
        return [qml.expval(qml.PauliZ(i)) for i in range(2)]
    return circuit

# --------------------------------------------------------------------------- #
# 3. Hybrid fraud detection model
# --------------------------------------------------------------------------- #

class FraudDetectionHybrid(nn.Module):
    """
    Hybrid model that uses a PennyLane variational circuit to generate
    quantum features from the classical input.  The quantum circuit is
    parameterised by `FraudLayerParameters` and produces a 2‑dimensional
    feature vector.  These quantum features are concatenated with the
    classical features and fed into a linear classifier.
    """
    def __init__(self, input_params: FraudLayerParameters,
                 layers: Iterable[FraudLayerParameters],
                 dev: qml.Device = qml.device("default.qubit", wires=2)):
        super().__init__()
        self.dev = dev
        # classical linear layers
        self.classical_layers: List[nn.Module] = []
        self.classical_layers.append(_layer_from_params(input_params, clip=False))
        self.classical_layers.extend(_layer_from_params(l, clip=True) for l in layers)
        self.classical_layers = nn.ModuleList(self.classical_layers)
        # quantum layers
        self.quantum_circuits: List[qml.QNode] = []
        self.quantum_circuits.append(_quantum_layer(input_params, self.dev))
        self.quantum_circuits.extend(_quantum_layer(l, self.dev) for l in layers)
        # final linear classifier
        self.final_linear = nn.Linear(4, 1)  # 2 classical + 2 quantum

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 2)
        for layer in self.classical_layers:
            x = layer(x)
        # quantum features
        q_feats = [qc(x) for qc in self.quantum_circuits]
        # stack outputs: each qc returns list of 2 expectations
        q_feats = torch.cat([torch.stack(q, dim=1) for q in q_feats], dim=1)
        # concatenate with classical features
        x = torch.cat([x, q_feats], dim=1)
        out = self.final_linear(x)
        return out

# --------------------------------------------------------------------------- #
# 4. Factory function
# --------------------------------------------------------------------------- #

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
    dev: qml.Device = qml.device("default.qubit", wires=2),
) -> nn.Module:
    """Return a FraudDetectionHybrid model."""
    return FraudDetectionHybrid(input_params, layers, dev)

__all__ = ["FraudLayerParameters", "build_fraud_detection_program",
           "FraudDetectionHybrid", "_layer_from_params"]
