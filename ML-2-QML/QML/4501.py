"""Quantum counterpart of HybridEstimatorQNN that builds a variational circuit with a swap‑test autoencoder and fraud‑style rotations."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler

class FraudLayerParameters:
    """Parameters that mirror the photonic fraud‑detection layer."""
    def __init__(self,
                 bs_theta: float, bs_phi: float,
                 phases: tuple[float, float],
                 squeeze_r: tuple[float, float],
                 squeeze_phi: tuple[float, float],
                 displacement_r: tuple[float, float],
                 displacement_phi: tuple[float, float],
                 kerr: tuple[float, float]) -> None:
        self.bs_theta = bs_theta
        self.bs_phi = bs_phi
        self.phases = phases
        self.squeeze_r = squeeze_r
        self.squeeze_phi = squeeze_phi
        self.displacement_r = displacement_r
        self.displacement_phi = displacement_phi
        self.kerr = kerr

class HybridEstimatorQNN:
    """Quantum estimator mirroring the classical HybridEstimatorQNN."""
    def __init__(self,
                 latent_dim: int = 3,
                 num_trash: int = 2,
                 fraud_params: FraudLayerParameters | None = None) -> None:
        self.latent_dim = latent_dim
        self.num_trash = num_trash
        self.fraud_params = fraud_params
        self.sampler = StatevectorSampler()
    def build_circuit(self) -> QuantumCircuit:
        qr = QuantumRegister(self.latent_dim + 2 * self.num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)
        # Variational ansatz for latent + trash qubits
        ansatz = RealAmplitudes(self.latent_dim + self.num_trash, reps=3)
        qc.compose(ansatz, range(self.latent_dim + self.num_trash), inplace=True)
        qc.barrier()
        # Swap‑test to read out a similarity signal
        aux = self.latent_dim + 2 * self.num_trash
        qc.h(aux)
        for i in range(self.num_trash):
            qc.cswap(aux, self.latent_dim + i, self.latent_dim + self.num_trash + i)
        qc.h(aux)
        qc.measure(aux, cr[0])
        # Fraud‑style parameterised rotations
        if self.fraud_params:
            for idx, phase in enumerate(self.fraud_params.phases):
                qc.rz(phase, idx)
            for idx, (r, phi) in enumerate(zip(self.fraud_params.squeeze_r, self.fraud_params.squeeze_phi)):
                qc.rx(r, idx)
        return qc
    def estimate(self, params: np.ndarray | None = None) -> float:
        qc = self.build_circuit()
        result = self.sampler.run(qc, shots=1024).result()
        counts = result.get_counts()
        return counts.get('1', 0) / 1024
    def __repr__(self) -> str:
        return f"<HybridEstimatorQNN latent_dim={self.latent_dim} num_trash={self.num_trash}>"

__all__ = ["HybridEstimatorQNN", "FraudLayerParameters"]
