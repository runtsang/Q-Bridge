from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler as Sampler


@dataclass
class FraudLayerParameters:
    """
    Parameters describing a single photonic or quantum layer.
    """
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


class FraudDetectionHybrid:
    """
    Quantum component of the hybrid fraud detection model.
    Builds a parameterized circuit resembling SamplerQNN and
    produces a probability distribution over fraud/not fraud.
    """

    def __init__(self):
        self.input_params = ParameterVector("input", 2)
        self.weight_params = ParameterVector("weight", 4)
        self.circuit = self._build_circuit()
        self.sampler = Sampler()
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=self.input_params,
            weight_params=self.weight_params,
            sampler=self.sampler,
        )

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        # Input rotations
        qc.ry(self.input_params[0], 0)
        qc.ry(self.input_params[1], 1)
        # Entanglement
        qc.cx(0, 1)
        # Weight rotations
        for i in range(4):
            target = i % 2
            qc.ry(self.weight_params[i], target)
        return qc

    def run(
        self,
        input_vals: Sequence[float],
        weight_vals: Sequence[float],
    ) -> np.ndarray:
        """
        Execute the sampler and return probability distribution.
        """
        probs = self.qnn(
            input_params=np.array(input_vals, dtype=np.float64),
            weight_params=np.array(weight_vals, dtype=np.float64),
        )
        return probs


__all__ = ["FraudLayerParameters", "FraudDetectionHybrid"]
