"""Variational quantum sampler that extends the original SamplerQNN with deeper entanglement layers."""
from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN as QSamplerQNN
from qiskit.primitives import StatevectorSampler


def SamplerQNN() -> QSamplerQNN:
    """
    Build a parameterized quantum circuit that samples from a 2‑qubit distribution.
    The circuit contains multiple layers of RX/RY rotations and CX entanglements,
    increasing expressive power over the original 4‑weight design.
    """
    inputs = ParameterVector("input", 2)
    weights = ParameterVector("weight", 6)  # 6 trainable weights for richer parameter space

    qc = QuantumCircuit(2)

    # Layer 1: input rotations
    qc.ry(inputs[0], 0)
    qc.ry(inputs[1], 1)
    qc.cx(0, 1)

    # Layer 2: parameterized rotations
    qc.ry(weights[0], 0)
    qc.ry(weights[1], 1)
    qc.cx(1, 0)

    # Layer 3: additional rotations
    qc.ry(weights[2], 0)
    qc.ry(weights[3], 1)
    qc.cx(0, 1)

    # Layer 4: final rotations
    qc.ry(weights[4], 0)
    qc.ry(weights[5], 1)

    # Optional additional entanglement for symmetry
    qc.cx(1, 0)

    sampler = StatevectorSampler()
    sampler_qnn = QSamplerQNN(
        circuit=qc,
        input_params=inputs,
        weight_params=weights,
        sampler=sampler,
    )
    return sampler_qnn


__all__ = ["SamplerQNN"]
