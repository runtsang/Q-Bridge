import pennylane as qml
import numpy as np
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass
class FraudLayerParameters:
    """Parameters for a single photonic layer in the quantum circuit."""
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


def _apply_layer(qc: qml.QNode, params: FraudLayerParameters, clip: bool = False) -> None:
    """Append a photonic layer to a PennyLane quantum circuit."""
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


def build_fraud_detection_qnode(
    dev: qml.Device,
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> qml.QNode:
    """Return a parametrised QNode that evaluates the hybrid quantum circuit."""
    @qml.qnode(dev, interface="torch")
    def circuit(inputs: torch.Tensor) -> torch.Tensor:
        # encode inputs as coherent displacements (one mode per feature)
        for i, inp in enumerate(inputs):
            qml.Dgate(inp, wires=i)
        # first photonic layer (unclipped)
        _apply_layer(circuit, input_params, clip=False)
        # subsequent layers
        for layer in layers:
            _apply_layer(circuit, layer, clip=True)
        # measurement: expectation of PauliZ on each mode
        return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

    return circuit


class FraudDetectionQuantumCircuit:
    """Class that encapsulates a variational photonic circuit with parameter‑shift gradients."""
    def __init__(
        self,
        dev: qml.Device,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
    ) -> None:
        self.circuit = build_fraud_detection_qnode(dev, input_params, layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run the circuit and return a linear combination of measurements."""
        z0, z1 = self.circuit(inputs)
        return 0.5 * (z0 + z1)  # simple classifier head

    def gradient(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compute parameter‑shift gradient of the circuit output."""
        return qml.gradients.param_shift(self.circuit)(inputs)

__all__ = ["FraudLayerParameters", "FraudDetectionQuantumCircuit"]
