"""Hybrid sampler‑classifier quantum circuit.

This module builds a parameterized quantum circuit that first encodes the
input data, then applies a layered ansatz followed by a sampling/measurement
stage.  The returned `SamplerQNN` instance can be used directly with
Qiskit Machine Learning primitives.

The design merges the data‑uploading scheme from the quantum classifier
and the sampling interface from the original `SamplerQNN` seed.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler


def build_sampler_classifier_circuit(
    num_qubits: int, depth: int
) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
    """Construct a layered ansatz with data‑uploading encoding and sampling observables.

    Parameters
    ----------
    num_qubits: int
        Number of qubits / input features.
    depth: int
        Number of variational layers.

    Returns
    -------
    circuit: QuantumCircuit
        Parameterized circuit ready for sampling.
    encoding: List[ParameterVector]
        List containing the input encoding parameters.
    weights: List[ParameterVector]
        List containing the variational parameters.
    observables: List[SparsePauliOp]
        Pauli‑Z observables used for sampling each qubit.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    qc = QuantumCircuit(num_qubits)

    # Data‑uploading encoding
    for qubit, param in enumerate(encoding):
        qc.rx(param, qubit)

    # Layered ansatz
    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            qc.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            qc.cz(qubit, qubit + 1)

    # Sampling observables: Z on each qubit
    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]

    return qc, [encoding], [weights], observables


def create_sampler_classifier_qnn(num_qubits: int, depth: int) -> SamplerQNN:
    """Instantiate a Qiskit `SamplerQNN` using the hybrid circuit.

    Parameters
    ----------
    num_qubits: int
        Number of qubits / input features.
    depth: int
        Number of variational layers.

    Returns
    -------
    SamplerQNN
        A Qiskit Machine Learning sampler ready for training.
    """
    circuit, encoding, weights, _ = build_sampler_classifier_circuit(num_qubits, depth)
    sampler = StatevectorSampler()
    return SamplerQNN(
        circuit=circuit,
        input_params=encoding[0],
        weight_params=weights[0],
        sampler=sampler,
    )


__all__ = ["build_sampler_classifier_circuit", "create_sampler_classifier_qnn"]
