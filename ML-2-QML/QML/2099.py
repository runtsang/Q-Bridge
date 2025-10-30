"""Quantum‑classical hybrid model for fraud detection.

The module extends the original photonic circuit to a PennyLane
parameterised QNode that can be trained end‑to‑end together with
classical layers.  It supports:
* Automatic clipping of parameters.
* A variational layer that mirrors the photonic construction.
* A full PyTorch‑compatible model class that mixes classical
  preprocessing with a quantum feature extractor.
* A lightweight training routine that uses autograd.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import pennylane as qml
import torch
from torch import nn
from torch.utils.data import DataLoader

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

    def __post_init__(self) -> None:
        """Clip all continuous parameters to safe ranges."""
        clip = lambda v: max(-5.0, min(5.0, v))
        self.bs_theta, self.bs_phi = clip(self.bs_theta), clip(self.bs_phi)
        self.phases = tuple(clip(p) for p in self.phases)
        self.squeeze_r = tuple(clip(r) for r in self.squeeze_r)
        self.squeeze_phi = tuple(clip(p) for p in self.squeeze_phi)
        self.displacement_r = tuple(clip(r) for r in self.displacement_r)
        self.displacement_phi = tuple(clip(p) for p in self.displacement_phi)
        self.kerr = tuple(max(-1.0, min(1.0, k)) for k in self.kerr)

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> qml.QNode:
    """
    Create a PennyLane QNode that implements the same layered
    photonic architecture as the original Strawberry Fields circuit.
    The QNode accepts a 2‑dimensional classical input and returns
    a single expectation value of the Pauli‑Z operator on the first wire.
    """
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Variational circuit mirroring the photonic construction."""
        # Encode the 2-dimensional input as rotations
        qml.RX(inputs[0], wires=0)
        qml.RX(inputs[1], wires=1)

        # Helper to unpack a flattened parameter vector
        def unpack(p: torch.Tensor) -> FraudLayerParameters:
            return FraudLayerParameters(
                bs_theta=float(p[0]),
                bs_phi=float(p[1]),
                phases=(float(p[2]), float(p[3])),
                squeeze_r=(float(p[4]), float(p[5])),
                squeeze_phi=(float(p[6]), float(p[7])),
                displacement_r=(float(p[8]), float(p[9])),
                displacement_phi=(float(p[10]), float(p[11])),
                kerr=(float(p[12]), float(p[13])),
            )

        # Apply the first (un‑clipped) layer
        _apply_layer(circuit, input_params, clip=False)

        # Apply each subsequent layer
        offset = 0
        for layer in layers:
            offset += 14  # number of parameters per layer
            _apply_layer(circuit, layer, clip=True)

        return qml.expval(qml.PauliZ(0))

    return circuit

def _apply_layer(circuit, params: FraudLayerParameters, *, clip: bool) -> None:
    """Helper that applies a single photonic layer to the circuit."""
    # Beam splitter gate
    theta = params.bs_theta if not clip else _clip(params.bs_theta, 5.0)
    phi = params.bs_phi if not clip else _clip(params.bs_phi, 5.0)
    qml.CNOT(wires=[0, 1])  # Emulate a 50/50 beam splitter via CNOT + Hadamard
    qml.Hadamard(wires=0)
    qml.Hadamard(wires=1)

    # Phase gates
    for i, phase in enumerate(params.phases):
        qml.PhaseShift(phase, wires=i)

    # Squeezing ≈ implemented with rotation‑gate pairs
    for i, (r, phi_s) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        r = r if not clip else _clip(r, 5.0)
        qml.Rot(r, 0.0, phi_s, wires=i)

    # Displacement ≈ implemented with Rx and Ry rotations
    for i, (r, phi_d) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        r = r if not clip else _clip(r, 5.0)
        qml.Rot(r, phi_d, 0.0, wires=i)

    # Kerr non‑linearity ≈ implemented with controlled‑phase shift
    for i, k in enumerate(params.kerr):
        k = k if not clip else _clip(k, 1.0)
        qml.ControlledPhaseShift(k, control=[i], wires=i)

class FraudQMLModel(nn.Module):
    """
    End‑to‑end hybrid model: classical preprocessing → quantum feature extractor →
    classical classifier.

    Parameters
    ----------
    input_params, layers:
        Photonic layer parameters for the quantum circuit.
    n_classes:
        Number of output classes (default 1 for binary fraud detection).
    """
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        n_classes: int = 1,
    ) -> None:
        super().__init__()
        self.qnode = build_fraud_detection_program(input_params, layers)
        self.classifier = nn.Linear(1, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 2)
        logits = []
        for xi in x:
            out = self.qnode(xi, torch.tensor([0.0]))  # Dummy parameter placeholder
            logits.append(out)
        logits = torch.stack(logits, dim=0)
        return self.classifier(logits)

def train_qml_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int = 20,
    lr: float = 1e-3,
    device: str = "cpu",
) -> nn.Module:
    """Training loop for the hybrid QML model."""
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x).squeeze(-1)
            loss = criterion(logits, y.float())
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            correct = 0
            total = 0
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x).squeeze(-1)
                val_loss += criterion(logits, y.float()).item()
                preds = logits > 0
                correct += preds.eq(y.byte()).sum().item()
                total += y.size(0)
            val_loss /= len(val_loader)
            val_acc = correct / total
        print(f"QML Epoch {epoch:02d} | Val loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

    return model

def evaluate_qml_model(
    model: nn.Module,
    data_loader: DataLoader,
    *,
    device: str = "cpu",
) -> Tuple[float, float]:
    """Return (average loss, accuracy) for the hybrid QML model."""
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x).squeeze(-1)
            total_loss += criterion(logits, y.float()).item()
            preds = logits > 0
            correct += preds.eq(y.byte()).sum().item()
            total += y.size(0)
    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total
    return avg_loss, accuracy

__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "FraudQMLModel",
    "train_qml_model",
    "evaluate_qml_model",
]
