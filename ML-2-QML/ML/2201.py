import torch
from torch import nn
import numpy as np
from typing import Iterable

# Import the quantum part from the QML module
from qml import FraudDetectorHybrid as QuantumFraudDetector

class FraudDetectorHybrid(nn.Module):
    """Hybrid fraud detection model that combines a classical neural network
    with quantum‑derived features.

    The architecture mirrors the classical FraudDetection model but augments
    the input with two quantum‑derived features obtained by evaluating a
    parametric Qiskit circuit.  The quantum circuit is defined in the
    companion QML module and its parameters are fixed at construction
    time, allowing the classical network to learn a mapping from
    (x1, x2, q1, q2) to the fraud score.
    """
    def __init__(
        self,
        input_params: object,
        layers: Iterable[object],
        clip: bool = True
    ) -> None:
        super().__init__()
        # Quantum feature extractor
        self.quantum_model = QuantumFraudDetector(
            input_params=input_params,
            layers=layers,
            clip=clip
        )
        # Classical network: input size 4 (2 classical + 2 quantum)
        self.classical_net = nn.Sequential(
            nn.Linear(4, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (batch, 2)
        with torch.no_grad():
            q_features = self.quantum_model.evaluate(
                inputs.detach().cpu().numpy()
            )
        q_features = torch.from_numpy(q_features).to(
            inputs.device, dtype=inputs.dtype
        )
        combined = torch.cat([inputs, q_features], dim=1)
        return self.classical_net(combined)

__all__ = ["FraudDetectorHybrid"]
