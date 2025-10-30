"""Quantum variational classifier with optional SamplerQNN output."""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler


def build_classifier_circuit(num_qubits: int, depth: int, use_sampler: bool = False):
    """
    Construct a layered ansatz for a hybrid quantum classifier.

    Args:
        num_qubits: Number of qubits (equal to the number of encoded features).
        depth: Number of variational layers.
        use_sampler: If True, returns a SamplerQNN object; otherwise returns the raw circuit.

    Returns:
        circuit (QuantumCircuit or SamplerQNN), encoding parameters, weight parameters, observables.
    """
    # Data encoding
    encoding = ParameterVector("x", num_qubits)
    # Variational parameters
    weights = ParameterVector("theta", num_qubits * depth)

    qc = QuantumCircuit(num_qubits)
    for qubit, param in enumerate(encoding):
        qc.rx(param, qubit)

    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            qc.ry(weights[idx], qubit)
            idx += 1
        # Entangling pattern
        for qubit in range(num_qubits - 1):
            qc.cz(qubit, qubit + 1)

    # Observables for a binary classifier
    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
                   for i in range(num_qubits)]

    if use_sampler:
        sampler = StatevectorSampler()
        sampler_qnn = SamplerQNN(
            circuit=qc,
            input_params=encoding,
            weight_params=weights,
            sampler=sampler
        )
        return sampler_qnn, list(encoding), list(weights), observables
    else:
        return qc, list(encoding), list(weights), observables


__all__ = ["build_classifier_circuit"]
