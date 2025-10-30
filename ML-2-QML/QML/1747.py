"""Quantum sampler network with a multi‑layer parameterised circuit."""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import Sampler as QiskitSampler
from qiskit.providers.aer import AerSimulator

def SamplerQNN():
    """
    Builds a quantum sampler that mirrors the classical architecture above.
    The circuit uses a two‑qubit feature map followed by two layers of
    parameterised rotations and entanglement. The sampler is executed on
    the Aer statevector backend for deterministic probabilities.
    """
    # Input feature parameters (2‑dimensional)
    inputs = ParameterVector("x", 2)

    # Weight parameters for the two rotation layers
    weights = ParameterVector("w", 8)

    qc = QuantumCircuit(2)

    # Feature map: encode classical inputs with Ry rotations
    qc.ry(inputs[0], 0)
    qc.ry(inputs[1], 1)

    # First entangling layer
    qc.cx(0, 1)

    # Parameterised rotation layer 1
    qc.ry(weights[0], 0)
    qc.rz(weights[1], 0)
    qc.ry(weights[2], 1)
    qc.rz(weights[3], 1)

    # Second entangling layer
    qc.cx(1, 0)

    # Parameterised rotation layer 2
    qc.ry(weights[4], 0)
    qc.rz(weights[5], 0)
    qc.ry(weights[6], 1)
    qc.rz(weights[7], 1)

    # Backend for simulation
    backend = AerSimulator(method="statevector")

    sampler = QiskitSampler(backend=backend)

    # Construct the Qiskit SamplerQNN with the circuit and parameter vectors
    sampler_qnn = SamplerQNN(
        circuit=qc,
        input_params=inputs,
        weight_params=weights,
        sampler=sampler,
    )

    return sampler_qnn

__all__ = ["SamplerQNN"]
