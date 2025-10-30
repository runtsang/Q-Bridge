import pennylane as qml
import numpy as np
import torch
from dataclasses import dataclass
from typing import Iterable, Tuple, List

@dataclass
class FraudLayerParameters:
    """Parameters describing a single photonic layer."""
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

def _apply_layer(params: FraudLayerParameters, clip: bool = False):
    """Return a function that applies a photonic layer using Pennylane ops."""
    def circuit():
        # Beam‑splitter
        qml.BSgate(params.bs_theta, params.bs_phi, wires=[0, 1])
        for i, phase in enumerate(params.phases):
            qml.Rgate(phase, wires=i)
        for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
            qml.Sgate(r if not clip else _clip(r, 5), phi, wires=i)
        qml.BSgate(params.bs_theta, params.bs_phi, wires=[0, 1])
        for i, phase in enumerate(params.phases):
            qml.Rgate(phase, wires=i)
        for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
            qml.Dgate(r if not clip else _clip(r, 5), phi, wires=i)
        for i, k in enumerate(params.kerr):
            qml.Kgate(k if not clip else _clip(k, 1), wires=i)
    return circuit

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters]
) -> qml.QNode:
    """Create a Pennylane QNode mirroring the photonic fraud‑detection program."""
    dev = qml.device("default.qubit", wires=2)
    @qml.qnode(dev, interface="torch")
    def circuit(inputs):
        # The input tensor is ignored; the circuit is parameterised by the layer params
        _apply_layer(input_params, clip=False)()
        for layer in layers:
            _apply_layer(layer, clip=True)()
        # Return expectation of Pauli‑Z on the first qubit as the output
        return qml.expval(qml.PauliZ(0))
    return circuit

class QuantumHybridHead:
    """Two‑qubit variational circuit that outputs an expectation value."""
    def __init__(self, shots: int = 1024):
        self.shots = shots
        self.dev = qml.device("default.qubit", wires=1)
        @qml.qnode(self.dev, interface="torch")
        def circuit(theta):
            qml.RY(theta, wires=0)
            return qml.expval(qml.PauliZ(0))
        self.circuit = circuit

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs shape: (batch, 1)
        return self.circuit(inputs)

class FraudDetectionHybridNet:
    """Hybrid photonic‑quantum fraud‑detection network."""
    def __init__(self,
                 input_params: FraudLayerParameters,
                 hidden_params: List[FraudLayerParameters],
                 use_quantum_head: bool = False):
        self.use_quantum_head = use_quantum_head
        self.circuit = build_fraud_detection_program(input_params, hidden_params)
        if self.use_quantum_head:
            self.head = QuantumHybridHead()
        else:
            self.head = None

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: torch.Tensor of shape (batch, 2)
        outputs = self.circuit(inputs)
        if self.use_quantum_head:
            outputs = self.head(outputs)
        probs = torch.sigmoid(outputs)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["FraudLayerParameters",
           "build_fraud_detection_program",
           "QuantumHybridHead",
           "FraudDetectionHybridNet"]
