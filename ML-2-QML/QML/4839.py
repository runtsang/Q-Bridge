"""Hybrid quantum estimator that mirrors the classical architecture."""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector, Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import Sampler


def _build_conv_circuit(kernel_size: int, threshold: float) -> tuple[QuantumCircuit, ParameterVector]:
    """
    Builds a data‑encoding subcircuit that maps classical pixel values
    into rotation angles.  Each qubit receives an rx gate with a parameter
    that will be bound to π if the corresponding pixel exceeds *threshold*,
    otherwise 0.0.  This mimics the thresholding used in the classical
    ConvFilter.
    """
    n_qubits = kernel_size ** 2
    circuit = QuantumCircuit(n_qubits)
    theta = ParameterVector("theta", n_qubits)
    for q in range(n_qubits):
        circuit.rx(theta[q], q)
    circuit.barrier()
    return circuit, theta


def _build_classifier_circuit(num_qubits: int, depth: int) -> tuple[QuantumCircuit, ParameterVector, ParameterVector, list[SparsePauliOp]]:
    """
    Constructs a layered ansatz: an initial data‑encoding layer (rx),
    followed by *depth* cycles of ry rotations and CX‑style entanglement.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("w", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    # Observables for a 2‑class regression target (one Z per qubit)
    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]
    return circuit, encoding, weights, observables


def EstimatorQNN() -> QiskitEstimatorQNN:
    """
    Factory that returns a Qiskit EstimatorQNN instance.

    The circuit is a composition of a data‑encoding conv subcircuit
    and a classification ansatz.  The input parameters correspond to
    the encoding qubits; the weight parameters are the variational
    ry angles.  The EstimatorQNN wraps a Sampler primitive for
    evaluation.
    """
    kernel_size = 2
    threshold = 127  # matches the classical ConvFilter threshold
    conv_circuit, conv_params = _build_conv_circuit(kernel_size, threshold)

    num_qubits = conv_circuit.num_qubits
    depth = 2
    classifier_circuit, enc_params, weight_params, observables = _build_classifier_circuit(num_qubits, depth)

    # Combine encoding and classifier into a single circuit
    full_circuit = conv_circuit.compose(classifier_circuit, front=True)

    sampler = Sampler()
    estimator = QiskitEstimatorQNN(
        circuit=full_circuit,
        observables=observables,
        input_params=enc_params,          # data‑encoding parameters
        weight_params=weight_params,      # variational parameters
        estimator=sampler,
    )
    return estimator


__all__ = ["EstimatorQNN"]
