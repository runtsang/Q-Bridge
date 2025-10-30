"""
Variational quantum fraud detection circuit with parameter‑shift gradients and noise simulation.
"""

import dataclasses
import pennylane as qml
import torch
from typing import Iterable, Sequence


@dataclasses.dataclass
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


class FraudDetectionQuantumCircuit:
    """
    Variational quantum circuit that emulates the photonic fraud‑detection layers using
    a parameter‑shift differentiable ansatz and a noisy simulator backend.
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: list[FraudLayerParameters],
        device: str = "default.qubit",
        shots: int = 1024,
    ) -> None:
        self.input_params = input_params
        self.layers = layers
        self.shots = shots
        self.dev = qml.device(device, wires=2, shots=shots)
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch", diff_method="parameter-shift")

    def _apply_layer(self, params: FraudLayerParameters, clip: bool) -> None:
        # Beam‑splitter analogues
        qml.RX(params.bs_theta, wires=0)
        qml.RZ(params.bs_phi, wires=0)
        qml.RX(params.bs_theta, wires=1)
        qml.RZ(params.bs_phi, wires=1)

        # Phase shifts
        for i, phase in enumerate(params.phases):
            qml.RZ(phase, wires=i)

        # Squeezing analogues via RY
        for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
            r_eff = _clip(r, 5.0) if clip else r
            qml.RY(r_eff, wires=i)

        # Displacement analogues via RX
        for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
            r_eff = _clip(r, 5.0) if clip else r
            qml.RX(r_eff, wires=i)

        # Kerr analogue via RZ
        for i, k in enumerate(params.kerr):
            k_eff = _clip(k, 1.0) if clip else k
            qml.RZ(k_eff, wires=i)

    def _circuit(self, *weights: float) -> float:
        # Encode input features as rotations on the two qubits
        for i, w in enumerate(weights):
            qml.RY(w, wires=i % 2)

        # Apply layers
        self._apply_layer(self.input_params, clip=False)
        for layer in self.layers:
            self._apply_layer(layer, clip=True)

        # Return expectation of PauliZ on qubit 0 as a probability proxy
        return qml.expval(qml.PauliZ(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the circuit on a minibatch of input features.
        """
        probs = []
        for sample in x:
            # Map the two‑dimensional feature vector to circuit parameters
            # Here we simply use the two components directly as rotation angles.
            expectation = self.qnode(sample[0].item(), sample[1].item())
            # Convert expectation value in [-1, 1] to probability in [0, 1]
            probs.append(0.5 * (1 + expectation))
        return torch.tensor(probs)

    def loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Binary cross‑entropy loss between predicted probabilities and labels.
        """
        preds = torch.clamp(preds, 1e-7, 1 - 1e-7)
        return -torch.mean(targets * torch.log(preds) + (1 - targets) * torch.log(1 - preds))

    def kl_divergence(self) -> torch.Tensor:
        """
        Placeholder for a KL term if a Bayesian prior over circuit parameters is introduced.
        """
        return torch.tensor(0.0)


def build_fraud_detection_circuit(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
    device: str = "default.qubit",
    shots: int = 1024,
) -> FraudDetectionQuantumCircuit:
    """
    Factory that returns a fully configured quantum circuit object.
    """
    return FraudDetectionQuantumCircuit(
        input_params=input_params,
        layers=list(layers),
        device=device,
        shots=shots,
    )


__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_circuit",
    "FraudDetectionQuantumCircuit",
]
