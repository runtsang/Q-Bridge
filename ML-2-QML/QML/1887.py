"""
Extended quantum sampler network.

This module builds a 4‑qubit variational circuit with two entangling layers
and parameterised Ry rotations.  The resulting `SamplerQNN` instance returns
probabilities over the 16 possible basis states, enabling richer sampling
behaviour than the 2‑qubit seed.
"""

from __future__ import annotations

from typing import List

from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN as QiskitSamplerQNN

# Parameters
NUM_QUBITS = 4
NUM_INPUT_PARAMS = NUM_QUBITS
NUM_WEIGHT_PARAMS = 2 * NUM_QUBITS  # two entangling layers


def SamplerQNN() -> QiskitSamplerQNN:
    """
    Construct and return a Qiskit `SamplerQNN` instance.

    The circuit consists of:
        1. Input Ry rotations on each qubit.
        2. First entangling layer of CX gates in a ring topology.
        3. Weight Ry rotations (first block).
        4. Second entangling layer (identical to the first).
        5. Weight Ry rotations (second block).
    The sampler uses a state‑vector backend to evaluate the circuit
    exactly, suitable for simulation or as a testbed for more advanced
    backend choices.
    """
    # Parameter vectors
    inputs = ParameterVector("input", NUM_INPUT_PARAMS)
    weights = ParameterVector("weight", NUM_WEIGHT_PARAMS)

    # Build the circuit
    qc = QuantumCircuit(NUM_QUBITS)
    # Input rotations
    for i in range(NUM_QUBITS):
        qc.ry(inputs[i], i)
    qc.barrier()

    # First entangling layer
    for i in range(NUM_QUBITS):
        qc.cx(i, (i + 1) % NUM_QUBITS)
    qc.barrier()

    # First block of weight rotations
    for i in range(NUM_QUBITS):
        qc.ry(weights[i], i)
    # Second entangling layer (identical to the first)
    for i in range(NUM_QUBITS):
        qc.cx(i, (i + 1) % NUM_QUBITS)
    qc.barrier()

    # Second block of weight rotations
    for i in range(NUM_QUBITS):
        qc.ry(weights[i + NUM_QUBITS], i)

    # Measure all qubits for a full basis‑state probability distribution
    qc.measure_all()

    # Sampler backend
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
