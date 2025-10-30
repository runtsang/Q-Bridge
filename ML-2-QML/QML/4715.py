"""Hybrid quantum sampler autoencoder classifier network."""
from __future__ import annotations

from typing import Iterable, Tuple, List

from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RealAmplitudes
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.quantum_info import SparsePauliOp


# --------------------------------------------------------------------------- #
# Circuit construction
# --------------------------------------------------------------------------- #
def build_hybrid_circuit(num_qubits: int,
                         depth_autoencoder: int = 5,
                         depth_classifier: int = 2) -> Tuple[
    QuantumCircuit,
    Iterable[ParameterVector],
    Iterable[ParameterVector],
    List[SparsePauliOp]
]:
    """
    Build a layered quantum circuit that first encodes data with a RealAmplitudes
    ansatz, then applies a classifier ansatz, and finally returns a list of
    Pauli observables for sampling.
    """
    # Parameter vectors
    encoding = ParameterVector("x", num_qubits)
    weights_auto = ParameterVector("theta", num_qubits * depth_autoencoder)
    weights_cls = ParameterVector("omega", num_qubits * depth_classifier)

    # Base circuit
    circuit = QuantumCircuit(num_qubits)

    # Data encoding with RealAmplitudes
    circuit.compose(RealAmplitudes(num_qubits, reps=depth_autoencoder),
                    qubits=range(num_qubits),
                    inplace=True)

    # Classifier layers
    idx = 0
    for _ in range(depth_classifier):
        for q in range(num_qubits):
            circuit.ry(weights_cls[idx], q)
            idx += 1
        for q in range(num_qubits - 1):
            circuit.cz(q, q + 1)

    # Observables: Z on each qubit
    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]

    return circuit, encoding, weights_cls, observables


# --------------------------------------------------------------------------- #
# Sampler wrapper
# --------------------------------------------------------------------------- #
def hybrid_sampler_qnn(num_qubits: int,
                       depth_autoencoder: int = 5,
                       depth_classifier: int = 2) -> SamplerQNN:
    """
    Instantiate a Qiskit SamplerQNN that samples the hybrid autoencoder+classifier
    ansatz. The input parameters are the data encoding; the weight parameters
    comprise the classifier variational angles.
    """
    circuit, encoding, weights, observables = build_hybrid_circuit(
        num_qubits, depth_autoencoder, depth_classifier
    )

    sampler = Sampler()
    qnn = SamplerQNN(
        circuit=circuit,
        input_params=encoding,
        weight_params=weights,
        observables=observables,
        sampler=sampler,
    )
    return qnn


__all__ = ["build_hybrid_circuit", "hybrid_sampler_qnn"]
