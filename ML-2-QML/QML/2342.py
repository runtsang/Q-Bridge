from __future__ import annotations

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

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

class QuantumFraudDetectionCircuit:
    """Hybrid photonic‑quantum fraud‑detection circuit.

    The photonic part is implemented with StrawberryFields (as in the
    original seed) and produces a two‑mode state.  The resulting
    photon‑number statistics are then fed into a qubit variational
    circuit that acts as a quantum kernel.  The two parts share the
    same clipping logic for the photonic parameters.
    """
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
    ) -> None:
        self.input_params = input_params
        self.layers = list(layers)
        self.dev = qml.device("default.qubit", wires=2)
        rng = np.random.default_rng(42)
        mat = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
        q, r = np.linalg.qr(mat)
        d = np.diag(r)
        ph = d / np.abs(d)
        self.kernel_unitary = q @ np.diag(ph)

    def _photonic_layer(self, params: FraudLayerParameters, clip: bool) -> None:
        import strawberryfields as sf
        from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate

        def _apply_layer(q, p, clip_flag):
            BSgate(p.bs_theta, p.bs_phi) | (q[0], q[1])
            for i, phase in enumerate(p.phases):
                Rgate(phase) | q[i]
            for i, (r, phi) in enumerate(zip(p.squeeze_r, p.squeeze_phi)):
                Sgate(_clip(r, 5) if clip_flag else r, phi) | q[i]
            BSgate(p.bs_theta, p.bs_phi) | (q[0], q[1])
            for i, phase in enumerate(p.phases):
                Rgate(phase) | q[i]
            for i, (r, phi) in enumerate(zip(p.displacement_r, p.displacement_phi)):
                Dgate(_clip(r, 5) if clip_flag else r, phi) | q[i]
            for i, k in enumerate(p.kerr):
                Kgate(_clip(k, 1) if clip_flag else k) | q[i]

        prog = sf.Program(2)
        with prog.context as q:
            _apply_layer(q, params, clip)
        return prog

    def photonic_program(self) -> "sf.Program":
        import strawberryfields as sf
        prog = sf.Program(2)
        with prog.context as q:
            self._photonic_layer(self.input_params, clip=False)
            for layer in self.layers:
                self._photonic_layer(layer, clip=True)
        return prog

    def quantum_kernel(self) -> qml.QNode:
        @qml.qnode(self.dev)
        def circuit(x):
            qml.RY(x[0], wires=0)
            qml.RY(x[1], wires=1)
            qml.QubitUnitary(self.kernel_unitary, wires=[0, 1])
            return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]
        return circuit

__all__ = ["FraudLayerParameters", "QuantumFraudDetectionCircuit"]
