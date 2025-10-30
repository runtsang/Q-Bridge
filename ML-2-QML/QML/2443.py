"""Quantum hybrid classifier with an autoencoder-inspired feature compression."""

from __future__ import annotations

from typing import Iterable, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import RealAmplitudes


def build_classifier_circuit(
    num_qubits: int,
    depth: int,
    latent_qubits: int = 3,
    trash_qubits: int = 2,
) -> Tuple[QuantumCircuit, Iterable, Iterable, list[SparsePauliOp]]:
    """
    Construct a layered variational circuit that first compresses the input
    into a low‑dimensional latent subspace via a RealAmplitudes autoencoder
    ansatz and then classifies with a second ansatz.
    """
    # Encoding parameters for classical data
    encoding = ParameterVector("x", num_qubits)

    # Autoencoder ansatz: RealAmplitudes on latent+trash qubits
    ae_qubits = latent_qubits + 2 * trash_qubits + 1
    ae_circuit = RealAmplitudes(ae_qubits, reps=5)

    # Classifier ansatz: RealAmplitudes on latent qubits
    clf_circuit = RealAmplitudes(latent_qubits, reps=depth)

    circuit = QuantumCircuit(num_qubits)

    # Data encoding
    for param, q in zip(encoding, range(num_qubits)):
        circuit.rx(param, q)

    # Embed input into first latent+trash qubits
    circuit.compose(ae_circuit, range(0, ae_qubits), inplace=True)

    # Swap test to project onto latent subspace
    aux = ae_qubits
    circuit.h(aux)
    for i in range(trash_qubits):
        circuit.cswap(aux, latent_qubits + i, latent_qubits + trash_qubits + i)
    circuit.h(aux)

    # Measurement of auxiliary qubit to collapse
    circuit.measure(aux, 0)

    # Apply classifier ansatz on latent qubits
    circuit.compose(clf_circuit, range(0, latent_qubits), inplace=True)

    # Observables on the latent qubits for a binary output
    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (latent_qubits - i - 1))
        for i in range(latent_qubits)
    ]

    # Weight parameters: ae params + clf params
    weights = list(ae_circuit.parameters) + list(clf_circuit.parameters)

    return circuit, list(encoding), weights, observables


__all__ = ["build_classifier_circuit"]
