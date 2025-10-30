"""Hybrid quantum sampler with fraud‑detection inspired rotations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler

@dataclass
class FraudLayerParameters:
    """Parameters describing a single photonic‑style layer."""
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

def build_hybrid_sampler_qnn(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> SamplerQNN:
    """
    Construct a parameterized quantum circuit that mimics the classical sampler
    and injects photonic‑style rotations for each fraud layer.
    """
    inputs = ParameterVector("input", 2)
    # Total weight parameters: 4 from base sampler + 14 per fraud layer
    total_weights = 4 + 14 * len(layers)
    weights = ParameterVector("weight", total_weights)

    qc = QuantumCircuit(2)
    idx = 0  # index into weights vector

    # Base sampler circuit
    qc.ry(weights[idx], 0); idx += 1
    qc.ry(weights[idx], 1); idx += 1
    qc.cx(0, 1)

    qc.ry(weights[idx], 0); idx += 1
    qc.ry(weights[idx], 1); idx += 1
    qc.cx(0, 1)

    qc.ry(weights[idx], 0); idx += 1
    qc.ry(weights[idx], 1); idx += 1

    # Fraud‑style layers
    for _ in layers:
        # Beam‑splitter rotations
        qc.ry(weights[idx], 0); idx += 1
        qc.ry(weights[idx], 1); idx += 1
        # Phase rotations
        qc.rz(weights[idx], 0); idx += 1
        qc.rz(weights[idx], 1); idx += 1
        # Squeezing rotations (approximated with Rx)
        qc.rx(weights[idx], 0); idx += 1
        qc.rx(weights[idx], 1); idx += 1
        # Squeezing phase rotations
        qc.rz(weights[idx], 0); idx += 1
        qc.rz(weights[idx], 1); idx += 1
        # Displacement rotations
        qc.rz(weights[idx], 0); idx += 1
        qc.rz(weights[idx], 1); idx += 1
        # Displacement phase rotations
        qc.rz(weights[idx], 0); idx += 1
        qc.rz(weights[idx], 1); idx += 1
        # Kerr rotations
        qc.rz(weights[idx], 0); idx += 1
        qc.rz(weights[idx], 1); idx += 1
        qc.cx(0, 1)

    # Instantiate the sampler
    sampler = StatevectorSampler()
    sampler_qnn = SamplerQNN(
        circuit=qc,
        input_params=inputs,
        weight_params=weights,
        sampler=sampler,
    )
    return sampler_qnn

__all__ = ["FraudLayerParameters", "build_hybrid_sampler_qnn"]
