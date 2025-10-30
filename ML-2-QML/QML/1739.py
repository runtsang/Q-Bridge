"""Quantum sampler network with a 3‑qubit parameterized ansatz and richer entanglement."""

from __future__ import annotations

from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit_machine_learning.neural_networks import SamplerQNN as QiskitSamplerQNN
from qiskit.primitives import StatevectorSampler

def SamplerQNN() -> QiskitSamplerQNN:
    """
    Construct a 3‑qubit variational sampler with additional entangling layers.
    Parameters:
        inputs: 3‑dimensional parameter vector (one per qubit)
        weights: 6‑dimensional weight vector (two per qubit)
    Returns:
        A qiskit_machine_learning SamplerQNN object ready for training.
    """
    # Define parameter vectors
    inputs = ParameterVector("input", 3)
    weights = ParameterVector("weight", 6)

    # Build a 3‑qubit ansatz
    qc = QuantumCircuit(3)

    # Rotation layers controlled by input parameters
    for q, inp in enumerate(inputs):
        qc.ry(inp, q)

    # Entangling pattern: linear chain with CX gates
    qc.cx(0, 1)
    qc.cx(1, 2)

    # Parameterized rotation layers
    for q, w in enumerate(weights[:3]):
        qc.ry(w, q)

    # Second entangling layer (full connectivity)
    for i in range(3):
        for j in range(i + 1, 3):
            qc.cx(i, j)

    # Final rotation layers
    for q, w in enumerate(weights[3:]):
        qc.ry(w, q)

    # Prepare a state‑vector sampler for efficient evaluation
    sampler = StatevectorSampler()

    # Instantiate the Qiskit SamplerQNN
    sampler_qnn = QiskitSamplerQNN(
        circuit=qc,
        input_params=inputs,
        weight_params=weights,
        sampler=sampler,
    )

    return sampler_qnn

__all__ = ["SamplerQNN"]
