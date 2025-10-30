"""Quantum circuit for fraud detection, designed to be used with Qiskit Machine Learning wrappers."""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RandomCircuit


def build_fraud_detection_circuit(num_qubits: int = 2, num_layers: int = 2) -> QuantumCircuit:
    """
    Construct a parameterized circuit that embeds input features via Ry gates
    and applies a series of random unitary layers.  The circuit is intended
    for use with estimators or samplers in Qiskit Machine Learning.

    Parameters
    ----------
    num_qubits : int
        Number of qubits (must match the dimensionality of the input vector).
    num_layers : int
        Number of random unitary layers to apply after the input encoding.

    Returns
    -------
    QuantumCircuit
        The fully parameterized circuit ready for binding inputs and weights.
    """
    # Input and weight parameters
    input_params = ParameterVector("input", num_qubits)
    weight_params = ParameterVector("weight", num_qubits * num_layers)

    qc = QuantumCircuit(num_qubits)

    # Input encoding with Ry gates
    for q in range(num_qubits):
        qc.ry(input_params[q], q)

    # Random unitary layers (depth 2 each)
    for l in range(num_layers):
        # Each layer receives its own set of weight parameters
        layer_circ = RandomCircuit(num_qubits, depth=2).decompose()
        # Bind the corresponding subset of weight parameters
        param_map = {f"theta_{i}": weight_params[l * num_qubits + i] for i in range(num_qubits)}
        layer_circ = layer_circ.bind_parameters(param_map)
        qc.append(layer_circ, range(num_qubits))

    return qc


__all__ = ["build_fraud_detection_circuit"]
