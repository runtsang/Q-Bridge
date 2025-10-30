"""Quantum fraud‑detection circuit with PennyLane hybrid interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pennylane as qml
import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate


@dataclass
class FraudLayerParameters:
    """Parameters for a single photonic layer, identical to the classical counterpart."""
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


def _apply_layer(
    q: Sequence, params: FraudLayerParameters, *, clip: bool
) -> None:
    """Apply a photonic layer to the Strawberry Fields program."""
    BSgate(params.bs_theta, params.bs_phi) | (q[0], q[1])
    for i, phase in enumerate(params.phases):
        Rgate(phase) | q[i]
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        Sgate(r if not clip else _clip(r, 5), phi) | q[i]
    BSgate(params.bs_theta, params.bs_phi) | (q[0], q[1])
    for i, phase in enumerate(params.phases):
        Rgate(phase) | q[i]
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        Dgate(r if not clip else _clip(r, 5), phi) | q[i]
    for i, k in enumerate(params.kerr):
        Kgate(k if not clip else _clip(k, 1)) | q[i]


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> sf.Program:
    """Create a Strawberry Fields program for the hybrid fraud detection model."""
    prog = sf.Program(2)
    with prog.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
    return prog


class FraudDetectionCircuit:
    """Hybrid quantum‑classical fraud‑detection interface.

    The class builds a PennyLane device backed by a Strawberry Fields simulator
    and exposes a variational circuit that can be trained with any PyTorch
    optimiser.
    """
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        *,
        device_name: str = "strawberryfields.fock",
        cutoff_dim: int = 8,
        shots: int = 1024,
    ) -> None:
        self.input_params = input_params
        self.layers = list(layers)
        self.device = qml.device(device_name, wires=2, cutoff_dim=cutoff_dim, shots=shots)
        self._build_variational_circuit()

    def _build_variational_circuit(self) -> None:
        """Construct a PennyLane QNode that evaluates the photonic circuit."""
        @qml.qnode(self.device, interface="torch")
        def circuit(*params):
            # params is a flattened list of all layer parameters in order
            idx = 0
            # first layer (no clipping)
            BSgate(params[idx], params[idx + 1]) | (0, 1)
            idx += 2
            for i in range(2):
                qml.RX(params[idx], wires=i)
                idx += 1
            for i in range(2):
                qml.RY(params[idx], wires=i)
                idx += 1
            for i in range(2):
                qml.RZ(params[idx], wires=i)
                idx += 1
            for i in range(2):
                qml.Rot(params[idx], params[idx + 1], params[idx + 2], wires=i)
                idx += 3
            # subsequent layers with clipping
            for layer in self.layers:
                BSgate(params[idx], params[idx + 1]) | (0, 1)
                idx += 2
                for i in range(2):
                    qml.RX(params[idx], wires=i)
                    idx += 1
                for i in range(2):
                    qml.RY(params[idx], wires=i)
                    idx += 1
                for i in range(2):
                    qml.RZ(params[idx], wires=i)
                    idx += 1
                for i in range(2):
                    qml.Rot(params[idx], params[idx + 1], params[idx + 2], wires=i)
                    idx += 3
            # Expectation of photon number on mode 0
            return qml.expval(qml.NumberOperator(0))

        self.circuit = circuit

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """Evaluate the circuit with a flattened parameter tensor."""
        return self.circuit(params)

    def get_flattened_params(self) -> torch.Tensor:
        """Return a flattened torch tensor of all parameters."""
        all_params = []
        for p in [self.input_params] + self.layers:
            all_params.extend([p.bs_theta, p.bs_phi])
            all_params.extend(p.phases)
            all_params.extend(p.squeeze_r)
            all_params.extend(p.squeeze_phi)
            all_params.extend(p.displacement_r)
            all_params.extend(p.displacement_phi)
            all_params.extend(p.kerr)
        return torch.tensor(all_params, dtype=torch.float32)

    def set_params_from_tensor(self, params: torch.Tensor) -> None:
        """Update internal parameter objects from a flattened tensor."""
        idx = 0
        def _slice(count):
            nonlocal idx
            slice_ = params[idx:idx+count]
            idx += count
            return slice_
        for p in [self.input_params] + self.layers:
            p.bs_theta, p.bs_phi = _slice(2).tolist()
            p.phases = tuple(_slice(2).tolist())
            p.squeeze_r = tuple(_slice(2).tolist())
            p.squeeze_phi = tuple(_slice(2).tolist())
            p.displacement_r = tuple(_slice(2).tolist())
            p.displacement_phi = tuple(_slice(2).tolist())
            p.kerr = tuple(_slice(2).tolist())

    def train_one_step(
        self,
        loss_fn: callable,
        optimizer: optim.Optimizer,
        target: torch.Tensor,
    ) -> float:
        """Perform a single gradient‑descent step on the circuit."""
        optimizer.zero_grad()
        pred = self.forward(self.get_flattened_params())
        loss = loss_fn(pred, target)
        loss.backward()
        optimizer.step()
        return loss.item()


__all__ = ["FraudLayerParameters", "FraudDetectionCircuit"]
