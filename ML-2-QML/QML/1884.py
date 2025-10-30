"""Quantum sampler network with a 4‑qubit entangled variational ansatz."""

from __future__ import annotations

from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit_machine_learning.neural_networks import SamplerQNN as QiskitSamplerQNN
from qiskit.primitives import StatevectorSampler as Sampler


def SamplerQNN() -> QiskitSamplerQNN:
    """
    Builds a 4‑qubit variational sampler circuit.
    The circuit uses a layered entangling pattern and
    separate parameter vectors for input and weight parameters.
    """
    # Define parameter vectors
    inputs = ParameterVector("input", 4)
    weights = ParameterVector("weight", 8)

    print(f"Input parameters: {[str(p) for p in inputs.params]}")
    print(f"Weight parameters: {[str(p) for p in weights.params]}")

    # Construct the circuit
    qc = QuantumCircuit(4)

    # Input encoding
    for i, p in enumerate(inputs):
        qc.ry(p, i)

    # Entangling layers
    for _ in range(2):
        for i in range(4):
            qc.ry(weights[_ * 2 + i], i)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 3)
        qc.cx(3, 0)

    # Final measurement layer
    qc.measure_all()

    # Visualise the circuit (optional)
    # qc.draw("mpl", style="clifford")

    # Create the sampler
    sampler = Sampler()
    sampler_qnn = QiskitSamplerQNN(
        circuit=qc,
        input_params=inputs,
        weight_params=weights,
        sampler=sampler,
    )
    return sampler_qnn


__all__ = ["SamplerQNN"]
