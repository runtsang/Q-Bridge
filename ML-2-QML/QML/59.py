import pennylane as qml
import numpy as np
from dataclasses import dataclass
from typing import Iterable, Sequence

@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

class FraudDetectionQuantumModel:
    """Quantum implementation of the fraud detection circuit using PennyLane.

    The model accepts 2‑dimensional classical inputs that are encoded as
    rotations on each qubit.  The circuit is parameterised by a flat
    vector that contains all beam‑splitter, phase, squeezing, displacement
    and Kerr parameters for every layer.  A single expectation value of
    :math:`Z\\otimes Z` is returned and interpreted as a fraud score.
    """
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        device: str = "default.qubit",
        shots: int | None = None,
    ) -> None:
        self.input_params = input_params
        self.layers = list(layers)
        self.device = qml.device(device, wires=2, shots=shots)
        self.params = self._pack_parameters()
        self.qnode = qml.QNode(self._circuit, self.device, interface="autograd")

    def _clip(self, value: float, bound: float) -> float:
        return max(-bound, min(bound, value))

    def _pack_parameters(self) -> np.ndarray:
        """Flatten the dataclasses into a 1‑D numpy array."""
        params = []
        for layer in [self.input_params] + self.layers:
            params.extend([layer.bs_theta, layer.bs_phi])
            params.extend(layer.phases)
            params.extend(layer.squeeze_r)
            params.extend(layer.squeeze_phi)
            params.extend(layer.displacement_r)
            params.extend(layer.displacement_phi)
            params.extend(layer.kerr)
        return np.array(params, dtype=np.float64)

    def _circuit(self, x0: float, x1: float, *param_values: float) -> float:
        """Quantum circuit mirroring the photonic architecture."""
        # Encode classical input as rotations
        qml.RY(x0, wires=0)
        qml.RY(x1, wires=1)

        ptr = 0

        def _apply_layer(clip: bool) -> None:
            nonlocal ptr
            bs_t = param_values[ptr]; bs_p = param_values[ptr+1]; ptr += 2
            phases = param_values[ptr:ptr+2]; ptr += 2
            squeeze_r = param_values[ptr:ptr+2]; ptr += 2
            squeeze_phi = param_values[ptr:ptr+2]; ptr += 2
            disp_r = param_values[ptr:ptr+2]; ptr += 2
            disp_phi = param_values[ptr:ptr+2]; ptr += 2
            kerr = param_values[ptr:ptr+2]; ptr += 2

            # Beam splitter (approximate with rotations)
            qml.RY(bs_t, wires=0)
            qml.RZ(bs_p, wires=0)
            qml.RY(bs_t, wires=1)
            qml.RZ(bs_p, wires=1)

            # Phase shifters
            for i, phase in enumerate(phases):
                qml.RZ(phase, wires=i)

            # Squeezing (approximated by a rotation pair)
            for i, (r, phi) in enumerate(zip(squeeze_r, squeeze_phi)):
                r_eff = self._clip(r, 5.0) if clip else r
                qml.RY(2 * np.arctan(np.exp(-r_eff)), wires=i)
                qml.RZ(phi, wires=i)

            # Displacement (approximated by a rotation pair)
            for i, (r, phi) in enumerate(zip(disp_r, disp_phi)):
                r_eff = self._clip(r, 5.0) if clip else r
                qml.RY(2 * np.arctan(np.exp(-r_eff)), wires=i)
                qml.RZ(phi, wires=i)

            # Kerr (non‑linear phase) – approximated by a RZ rotation
            for i, k in enumerate(kerr):
                k_eff = self._clip(k, 1.0) if clip else k
                qml.RZ(k_eff, wires=i)

        # First layer (fixed)
        _apply_layer(clip=False)
        # Subsequent trainable layers
        for _ in self.layers:
            _apply_layer(clip=True)

        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Evaluate the circuit for a batch of 2‑D inputs."""
        outputs = []
        for x in X:
            outputs.append(self.qnode(x[0], x[1], *self.params))
        return np.array(outputs)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lr: float = 0.01,
        epochs: int = 100,
    ) -> np.ndarray:
        """Mini‑batch gradient descent on the quantum circuit."""
        opt = qml.GradientDescentOptimizer(stepsize=lr)
        params = self.params
        for epoch in range(epochs):
            def loss_fn(p: np.ndarray) -> float:
                preds = np.array([self.qnode(x[0], x[1], *p) for x in X])
                return np.mean((preds - y) ** 2)
            params = opt.step(loss_fn, params)
            loss = loss_fn(params)
            print(f"Epoch {epoch+1:3d}/{epochs:3d}  loss={loss:.4f}")
        self.params = params
        return params

__all__ = ["FraudLayerParameters", "FraudDetectionQuantumModel"]
