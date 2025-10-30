"""Quantum circuit factory for depth‑controlled variational classifier."""

from __future__ import annotations

from typing import Iterable, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.primitives import StatevectorSampler


def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, list[SparsePauliOp]]:
    """
    Construct a layered ansatz with explicit encoding and variational parameters.
    Encoding parameters represent classical latent features.
    """
    # Classical encoding of latent vector
    encoding = ParameterVector("x", num_qubits)
    # Variational weights
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    # Encode classical latent features via RY rotations
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    # Variational depth layers
    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        # Entangling CZ chain
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    # Observables: Z on each qubit
    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return circuit, list(encoding), list(weights), observables


def QuantumAutoencoderClassifier(num_qubits: int, depth: int) -> SamplerQNN:
    """
    Returns a QNN that maps a classical latent vector (size=num_qubits)
    to a vector of expectation values via the variational circuit.
    """
    circuit, latent_params, weight_params, observables = build_classifier_circuit(num_qubits, depth)
    sampler = StatevectorSampler()
    qnn = SamplerQNN(
        circuit=circuit,
        input_params=latent_params,
        weight_params=weight_params,
        interpret=lambda x: x,  # identity
        output_shape=len(observables),
        sampler=sampler,
    )
    return qnn


__all__ = ["build_classifier_circuit", "QuantumAutoencoderClassifier"]
