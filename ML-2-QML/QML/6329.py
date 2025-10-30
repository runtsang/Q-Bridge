"""Quantum sampler network with a parameterized 4‑qubit circuit.

The circuit encodes two input parameters and four trainable weights.  It is
structured similarly to the original `SamplerQNN` example but expands the
parameter count and includes a small random layer to increase expressivity.
"""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN as QiskitSamplerQNN
from qiskit.primitives import StatevectorSampler


def SamplerQNN() -> QiskitSamplerQNN:
    """
    Construct a `SamplerQNN` instance using a 4‑qubit parameterized circuit.

    Returns:
        A `SamplerQNN` object ready for training or inference.
    """
    # Parameter vectors
    inputs = ParameterVector("input", 2)
    weights = ParameterVector("weight", 4)

    # 4‑qubit circuit
    qc = QuantumCircuit(4)

    # Encode inputs with Ry rotations on the first two qubits
    qc.ry(inputs[0], 0)
    qc.ry(inputs[1], 1)

    # Simple entangling layer
    qc.cx(0, 2)
    qc.cx(1, 3)

    # Random layer: alternating Ry on each qubit
    for i, w in enumerate(weights):
        qc.ry(w, i % 4)

    # Final entanglement
    qc.cx(2, 3)

    # Sampler primitive
    sampler = StatevectorSampler()

    # Build the Qiskit sampler QNN
    qnn = QiskitSamplerQNN(
        circuit=qc,
        input_params=inputs,
        weight_params=weights,
        sampler=sampler,
    )
    return qnn


__all__ = ["SamplerQNN"]
