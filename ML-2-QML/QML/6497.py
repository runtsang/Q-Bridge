"""Quantum counterpart of the fraud‑detection model using Qiskit.

The class mirrors the classical `FraudDetectionModel` interface but
builds a parameter‑shifted ansatz that encodes the same layer parameters.
A Bayesian prior is provided for all parameters, and a `predict` method
runs a QASM simulator to estimate the fraud probability.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple, List, Any

# --------------------------------------------------------------------------- #
# 1.  Hyper‑parameter definition (identical to classical)
# --------------------------------------------------------------------------- #
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

# --------------------------------------------------------------------------- #
# 2.  Quantum model
# --------------------------------------------------------------------------- #
class FraudDetectionModel:
    """A Qiskit implementation that emulates the photonic layer structure."""
    def __init__(self,
                 input_params: FraudLayerParameters,
                 layers: Iterable[FraudLayerParameters],
                 clip: bool = True) -> None:
        self.input_params = input_params
        self.layers_params = list(layers)
        self.clip = clip
        self.circuit = self._build_circuit()

    # --------------------------------------------------------------------- #
    # 3.  Circuit construction
    # --------------------------------------------------------------------- #
    def _clip(self, value: float, bound: float) -> float:
        return max(-bound, min(bound, value))

    def _apply_layer(self, params: FraudLayerParameters, clip_flag: bool) -> List[Any]:
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(2)
        # Beam‑splitter analogue: use RY rotations
        qc.ry(params.bs_theta, 0)
        qc.ry(params.bs_theta, 1)
        qc.rz(params.bs_phi, 0)
        qc.rz(params.bs_phi, 1)

        for i, phase in enumerate(params.phases):
            qc.rz(phase, i)

        for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
            r_val = self._clip(r, 5) if clip_flag else r
            qc.u3(r_val, phi, 0, i)  # generic gate for squeezing

        for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
            r_val = self._clip(r, 5) if clip_flag else r
            qc.rx(r_val, i)  # displacement analogue

        for i, k in enumerate(params.kerr):
            k_val = self._clip(k, 1) if clip_flag else k
            qc.rz(k_val, i)  # Kerr analogue

        return qc

    def _build_circuit(self):
        from qiskit import QuantumCircuit, ClassicalRegister
        qc = QuantumCircuit(2, 1)
        # Input layer
        qc += self._apply_layer(self.input_params, clip_flag=False)
        # Hidden layers
        for layer in self.layers_params:
            qc += self._apply_layer(layer, clip_flag=True)
        # Measurement
        qc.measure(0, 0)
        return qc

    # --------------------------------------------------------------------- #
    # 4.  Bayesian sampling
    # --------------------------------------------------------------------- #
    def sample_hyperparameters(self, rng: np.random.Generator | None = None) -> None:
        rng = rng or np.random.default_rng()
        def rand_param() -> FraudLayerParameters:
            return FraudLayerParameters(
                bs_theta=rng.uniform(-np.pi, np.pi),
                bs_phi=rng.uniform(-np.pi, np.pi),
                phases=(rng.uniform(-np.pi, np.pi), rng.uniform(-np.pi, np.pi)),
                squeeze_r=(rng.uniform(0, 5), rng.uniform(0, 5)),
                squeeze_phi=(rng.uniform(-np.pi, np.pi), rng.uniform(-np.pi, np.pi)),
                displacement_r=(rng.uniform(0, 5), rng.uniform(0, 5)),
                displacement_phi=(rng.uniform(-np.pi, np.pi), rng.uniform(-np.pi, np.pi)),
                kerr=(rng.uniform(-1, 1), rng.uniform(-1, 1)),
            )
        self.input_params = rand_param()
        self.layers_params = [rand_param() for _ in range(len(self.layers_params))]
        self.circuit = self._build_circuit()

    # --------------------------------------------------------------------- #
    # 5.  Execution & prediction
    # --------------------------------------------------------------------- #
    def run(self, backend=None, shots: int = 1024):
        from qiskit import Aer, execute
        backend = backend or Aer.get_backend('qasm_simulator')
        job = execute(self.circuit, backend=backend, shots=shots)
        result = job.result()
        counts = result.get_counts()
        # probability of measuring '1'
        prob = counts.get('1', 0) / shots
        return prob

    def predict(self, backend=None, shots: int = 1024) -> float:
        return self.run(backend=backend, shots=shots)

# --------------------------------------------------------------------------- #
# 6.  Public API
# --------------------------------------------------------------------------- #
__all__ = ["FraudLayerParameters", "FraudDetectionModel"]
