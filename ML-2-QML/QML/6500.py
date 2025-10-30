"""Quantum estimator that mirrors the classical fraud‑layer architecture.

The circuit is constructed from a list of FraudLayerParameters.  Each
parameter set drives a sequence of single‑qubit rotations and two‑qubit
entangling gates.  The first layer is unregularised, subsequent layers
enforce a clipping bound on rotation angles.  The class exposes a
qiskit_machine_learning.neural_networks.EstimatorQNN instance that
can be used directly in a variational optimisation loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QMLEstimatorQNN
from qiskit.primitives import Estimator as QiskitEstimator

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

class EstimatorQNN:
    """Quantum neural network with fraud‑layer inspired parameterisation."""
    def __init__(self, layers: List[FraudLayerParameters]) -> None:
        self.layers = layers
        self.circuit, self.input_params, self.weight_params = self._build_circuit(layers)
        self.estimator = QiskitEstimator()
        self.qnn = QMLEstimatorQNN(
            circuit=self.circuit,
            observables=self._observable(),
            input_params=self.input_params,
            weight_params=self.weight_params,
            estimator=self.estimator,
        )

    def _build_circuit(self, layers: List[FraudLayerParameters]) -> tuple[QuantumCircuit, List[Parameter], List[Parameter]]:
        qc = QuantumCircuit(2)
        input_params: List[Parameter] = []
        weight_params: List[Parameter] = []

        for idx, params in enumerate(layers):
            clip = idx > 0
            # Entangling gate analogous to BSgate
            theta = Parameter(f"theta_{idx}")
            phi = Parameter(f"phi_{idx}")
            qc.cx(0, 1)
            qc.ry(theta, 0)
            qc.rz(phi, 1)
            input_params.append(theta)
            weight_params.append(phi)

            # Single‑qubit rotations mimicking displacement and squeezing
            for i, (r, phi_s) in enumerate(zip(params.displacement_r, params.displacement_phi)):
                angle = Parameter(f"disp_{idx}_{i}")
                qc.rx(angle, i)
                weight_params.append(angle)
                if clip:
                    qc.rx(_clip(r, 5.0), i)  # fixed value for regularisation

            for i, (r, phi_s) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
                angle = Parameter(f"squeeze_{idx}_{i}")
                qc.rz(angle, i)
                weight_params.append(angle)
                if clip:
                    qc.rz(_clip(r, 5.0), i)

            # Kerr‑like non‑linearity via RZZ
            for i, k in enumerate(params.kerr):
                angle = Parameter(f"kerr_{idx}_{i}")
                qc.rzz(angle, i, (i + 1) % 2)
                weight_params.append(angle)
                if clip:
                    qc.rzz(_clip(k, 1.0), i, (i + 1) % 2)

        qc.barrier()
        return qc, input_params, weight_params

    def _observable(self) -> SparsePauliOp:
        # Expectation value of Z on qubit 0
        return SparsePauliOp.from_list([("Z" * self.circuit.num_qubits, 1)])

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Return expectation values for a batch of classical inputs."""
        return self.estimator.run(
            self.circuit,
            inputs,
            observables=self._observable()
        ).values

__all__ = ["EstimatorQNN", "FraudLayerParameters"]
