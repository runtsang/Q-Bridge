"""Quantum sampler network using a 3‑qubit variational circuit."""
from __future__ import annotations

from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN as QSamplerQNN
from qiskit.primitives import StatevectorSampler


def SamplerQNN() -> QSamplerQNN:
    # Input and weight parameters
    inputs = ParameterVector("input", 3)
    weights = ParameterVector("weight", 6)

    # Build a 3‑qubit entangled variational circuit
    qc = QuantumCircuit(3)
    # Input rotation layer
    for i in range(3):
        qc.ry(inputs[i], i)

    # Entangling layer 1
    qc.cx(0, 1)
    qc.cx(1, 2)

    # Parameterized rotation layer
    for i in range(3):
        qc.ry(weights[i], i)

    # Entangling layer 2
    qc.cx(0, 1)
    qc.cx(1, 2)

    # Second parameterized rotation layer
    for i in range(3, 6):
        qc.ry(weights[i], i - 3)

    # Use a state‑vector sampler to obtain measurement probabilities
    sampler = StatevectorSampler()
    sampler_qnn = QSamplerQNN(circuit=qc,
                              input_params=inputs,
                              weight_params=weights,
                              sampler=sampler)
    return sampler_qnn


__all__ = ["SamplerQNN"]
