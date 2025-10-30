"""Quantum circuit builder and sampler that complements the classical classifier.

The circuit follows the same interface as the original seed but adds a feature‑map
layer that uses RX rotations for data encoding and a depth‑controlled variational
ansatz.  It also provides a convenience SamplerQNN helper that can be run on a
Qiskit backend or simulator.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import SamplerQNN as QiskitSamplerQNN
from qiskit.primitives import StatevectorSampler as Sampler


def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """Construct a layered ansatz with explicit encoding and variational parameters.

    Parameters
    ----------
    num_qubits : int
        Number of qubits / input features.
    depth : int
        Number of variational blocks.

    Returns
    -------
    circuit : QuantumCircuit
        The constructed circuit.
    encoding : list[Parameter]
        List of data‑encoding parameters.
    weights : list[Parameter]
        List of variational parameters.
    observables : list[SparsePauliOp]
        Pauli‑Z observables on each qubit, matching the classical output logits.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    qc = QuantumCircuit(num_qubits)
    # data encoding
    for qubit, param in enumerate(encoding):
        qc.rx(param, qubit)

    # variational layers
    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            qc.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            qc.cz(qubit, qubit + 1)

    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]

    return qc, list(encoding), list(weights), observables


def SamplerQNN() -> QiskitSamplerQNN:
    """Return a Qiskit SamplerQNN instance that mirrors the classical sampler.

    The returned object can be executed on any backend that supports state‑vector
    sampling, e.g. the Aer simulator.  It uses a 2‑qubit circuit with a simple
    CX‑based entanglement followed by parameterized RY rotations.
    """
    inputs = ParameterVector("input", 2)
    weights = ParameterVector("weight", 4)

    qc = QuantumCircuit(2)
    qc.ry(inputs[0], 0)
    qc.ry(inputs[1], 1)
    qc.cx(0, 1)
    qc.ry(weights[0], 0)
    qc.ry(weights[1], 1)
    qc.cx(0, 1)
    qc.ry(weights[2], 0)
    qc.ry(weights[3], 1)

    sampler = Sampler()
    return QiskitSamplerQNN(
        circuit=qc,
        input_params=inputs,
        weight_params=weights,
        sampler=sampler,
    )


__all__ = ["build_classifier_circuit", "SamplerQNN"]
