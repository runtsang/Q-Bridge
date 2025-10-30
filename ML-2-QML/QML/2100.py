"""Quantum‑variational fraud‑detection model using PennyLane.

The circuit reproduces the photonic layer structure of the seed
but is expressed in terms of PennyLane operations so it can be
trained with gradient descent.  Early stopping and parameter
clipping are also provided.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple, Sequence
import pennylane as qml
import numpy as np
import torch


@dataclass
class FraudLayerParameters:
    """Parameters describing a photonic layer."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]


def _clip(v: float, bound: float) -> float:
    return max(-bound, min(bound, v))


def _layer_params_to_vector(params: FraudLayerParameters) -> np.ndarray:
    """Flatten a layer into a 1‑D numpy array."""
    return np.array(
        [
            params.bs_theta,
            params.bs_phi,
            *params.phases,
            *params.squeeze_r,
            *params.squeeze_phi,
            *params.displacement_r,
            *params.displacement_phi,
            *params.kerr,
        ],
        dtype=np.float64,
    )


def _vector_to_layer_params(vec: np.ndarray) -> FraudLayerParameters:
    bs_theta, bs_phi = vec[0], vec[1]
    phases = tuple(vec[2:4])
    squeeze_r = tuple(vec[4:6])
    squeeze_phi = tuple(vec[6:8])
    displacement_r = tuple(vec[8:10])
    displacement_phi = tuple(vec[10:12])
    kerr = tuple(vec[12:14])
    return FraudLayerParameters(
        bs_theta, bs_phi, phases, squeeze_r, squeeze_phi,
        displacement_r, displacement_phi, kerr
    )


def _apply_layer(q, params: FraudLayerParameters, clip: bool) -> None:
    """Add a photonic layer to the circuit."""
    # Beam splitter
    qml.BSgate(params.bs_theta, params.bs_phi, wires=[0, 1])
    for i, phase in enumerate(params.phases):
        qml.Rgate(phase, wires=i)
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        r_val = _clip(r, 5) if clip else r
        qml.Sgate(r_val, phi, wires=i)
    qml.BSgate(params.bs_theta, params.bs_phi, wires=[0, 1])
    for i, phase in enumerate(params.phases):
        qml.Rgate(phase, wires=i)
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        r_val = _clip(r, 5) if clip else r
        qml.Dgate(r_val, phi, wires=i)
    for i, k in enumerate(params.kerr):
        k_val = _clip(k, 1) if clip else k
        # Kerr is approximated with an RX gate in qubit representation
        qml.RX(k_val, wires=i)


def _flatten_params(layers: Iterable[FraudLayerParameters]) -> np.ndarray:
    vecs = [_layer_params_to_vector(lp) for lp in layers]
    return np.concatenate(vecs)


def _unflatten_params(vec: np.ndarray, n_layers: int) -> List[FraudLayerParameters]:
    return [_vector_to_layer_params(vec[i * 14 : (i + 1) * 14]) for i in range(n_layers)]


class FraudDetectionHybrid:
    """Quantum‑variational fraud‑detection model."""

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        device_name: str = "default.qubit",
        wires: int = 2,
        clip: bool = True,
    ) -> None:
        self.input_params = input_params
        self.layers = list(layers)
        self.clip = clip
        self.n_layers = len(self.layers) + 1  # include input layer
        self.params_vector = torch.tensor(
            _flatten_params([input_params] + self.layers),
            dtype=torch.float64,
        )
        self.dev = qml.device(device_name, wires=wires)
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch")

    def _circuit(self, params_flat: torch.Tensor) -> torch.Tensor:
        """Quantum circuit that returns the expectation of PauliZ on qubit 0."""
        params_np = params_flat.detach().cpu().numpy()
        layers = _unflatten_params(params_np, self.n_layers)
        for layer in layers:
            _apply_layer(qml, layer, clip=self.clip)
        return qml.expval(qml.PauliZ(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map circuit expectation to a probability."""
        logits = self.qnode(self.params_vector)
        return torch.sigmoid(logits)

    def fit(
        self,
        train_loader,
        val_loader,
        epochs: int = 30,
        lr: float = 1e-2,
        early_stop_patience: int = 5,
    ) -> List[float]:
        """Train the variational circuit with binary cross‑entropy."""
        opt = torch.optim.Adam([self.params_vector], lr=lr)
        criterion = torch.nn.BCELoss()
        best_val = float("inf")
        patience = 0
        val_losses: List[float] = []

        for epoch in range(epochs):
            self.dev.update_options(shots=1024)
            for x, y in train_loader:
                opt.zero_grad()
                preds = self.forward(x)
                loss = criterion(preds.squeeze(), y.float())
                loss.backward()
                opt.step()

            self.dev.update_options(shots=4096)
            val_loss = 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    preds = self.forward(x)
                    val_loss += criterion(preds.squeeze(), y.float()).item()
            val_loss /= len(val_loader)
            val_losses.append(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                torch.save(self.params_vector.detach().cpu(), "best_fraud_params.pt")
                patience = 0
            else:
                patience += 1

            if patience >= early_stop_patience:
                print(f"Early stopping at epoch {epoch}")
                break

        self.params_vector = torch.load("best_fraud_params.pt")
        return val_losses

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return binary predictions."""
        self.dev.update_options(shots=8192)
        with torch.no_grad():
            probs = self.forward(x)
        return (probs > 0.5).float()

    @staticmethod
    def load_from_checkpoint(path: str) -> "FraudDetectionHybrid":
        """Instantiate from a saved parameter vector."""
        vec = torch.load(path)
        dummy = FraudLayerParameters(
            0, 0, (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)
        )
        return FraudDetectionHybrid(dummy, [], clip=False)

__all__ = ["FraudLayerParameters", "FraudDetectionHybrid"]
