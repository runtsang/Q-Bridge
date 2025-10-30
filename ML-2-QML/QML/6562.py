"""Quantum sampler network mirroring the classical fraud‑detection inspired architecture."""

from __future__ import annotations

from dataclasses import dataclass
from qiskit import QuantumCircuit, ParameterVector
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN as QSamplerQNN

@dataclass
class FraudLayerParameters:
    """Parameters describing a single photonic layer (used for quantum mapping)."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

class SamplerQNN:
    """Quantum sampler network that mirrors the classical fraud‑detection inspired structure."""
    def __init__(self, fraud_params: FraudLayerParameters):
        # Parameter vectors
        self.input_params = ParameterVector("input", 2)
        self.weight_params = ParameterVector("weight", 4)
        # Build the circuit
        self.circuit = self._build_circuit(fraud_params)
        # Sampler primitive
        self.sampler = Sampler()
        # Qiskit SamplerQNN
        self.qsampler = QSamplerQNN(
            circuit=self.circuit,
            input_params=self.input_params,
            weight_params=self.weight_params,
            sampler=self.sampler,
        )

    def _build_circuit(self, fraud_params: FraudLayerParameters) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        # Input rotations
        qc.ry(self.input_params[0], 0)
        qc.ry(self.input_params[1], 1)
        qc.cx(0, 1)
        # Weight rotations
        qc.ry(self.weight_params[0], 0)
        qc.ry(self.weight_params[1], 1)
        qc.cx(0, 1)
        qc.ry(self.weight_params[2], 0)
        qc.ry(self.weight_params[3], 1)
        # Fraud layer
        self._apply_fraud_layer(qc, fraud_params)
        return qc

    def _apply_fraud_layer(self, qc: QuantumCircuit, params: FraudLayerParameters) -> None:
        # bs_theta, bs_phi as RZ on each qubit
        qc.rz(params.bs_theta, 0)
        qc.rz(params.bs_phi, 1)
        # phases as RZ
        qc.rz(params.phases[0], 0)
        qc.rz(params.phases[1], 1)
        # squeeze_r and squeeze_phi as RZ
        qc.rz(params.squeeze_r[0], 0)
        qc.rz(params.squeeze_phi[0], 0)
        qc.rz(params.squeeze_r[1], 1)
        qc.rz(params.squeeze_phi[1], 1)
        # displacement_r and displacement_phi as RZ
        qc.rz(params.displacement_r[0], 0)
        qc.rz(params.displacement_phi[0], 0)
        qc.rz(params.displacement_r[1], 1)
        qc.rz(params.displacement_phi[1], 1)
        # kerr as RZ
        qc.rz(params.kerr[0], 0)
        qc.rz(params.kerr[1], 1)

    def forward(self, x):
        """Proxy the Qiskit SamplerQNN forward method."""
        return self.qsampler.forward(x)

__all__ = ["SamplerQNN", "FraudLayerParameters"]
