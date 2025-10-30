"""Quantum circuit builder for fraud detection, combining encoding, variational layers, and observables."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.random import random_circuit


def build_fraud_detection_circuit(
    num_qubits: int,
    depth: int,
) -> Tuple[QuantumCircuit, Iterable, Iterable, list[SparsePauliOp]]:
    """
    Construct a layered ansatz with data encoding and a random variational backbone.
    Parameters
    ----------
    num_qubits : int
        Number of qubits representing the feature vector.
    depth : int
        Depth of the variational layers.
    Returns
    -------
    circuit : QuantumCircuit
        The full circuit ready for simulation or execution.
    encoding : list[Parameter]
        Parameters used for data encoding.
    weights : list[Parameter]
        Variational parameters.
    observables : list[SparsePauliOp]
        Observable set for the final measurement.
    """
    # Data encoding
    encoding = ParameterVector("x", num_qubits)
    # Variational parameters
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)

    # Encode data in X‑rotations
    for idx, qubit in enumerate(range(num_qubits)):
        circuit.rx(encoding[idx], qubit)

    # Variational layers
    param_idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[param_idx], qubit)
            param_idx += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    # Add a random circuit to enrich expressivity
    circuit += random_circuit(num_qubits, 2)

    # Measurement
    circuit.measure_all()

    # Observables: single‑qubit Z on each qubit
    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]

    return circuit, list(encoding), list(weights), observables


def run_fraud_detection_circuit(
    circuit: QuantumCircuit,
    data: np.ndarray,
    shots: int = 1024,
    backend: Optional[qiskit.providers.Backend] = None,
) -> np.ndarray:
    """
    Execute the fraud‑detection circuit on a batch of classical data.
    Parameters
    ----------
    circuit : QuantumCircuit
        The circuit built by ``build_fraud_detection_circuit``.
    data : np.ndarray
        2‑D array of shape (batch, num_qubits) with values in [0, 1].
    shots : int
        Number of shots per execution.
    backend : Backend, optional
        Qiskit backend; defaults to Aer qasm simulator.
    Returns
    -------
    np.ndarray
        Expectation values of the observables for each data point.
    """
    if backend is None:
        backend = qiskit.Aer.get_backend("qasm_simulator")

    # Bind data to encoding parameters
    param_binds = [
        {circuit.parameters[i]: np.pi * val for i, val in enumerate(row)}
        for row in data
    ]

    job = qiskit.execute(
        circuit,
        backend,
        shots=shots,
        parameter_binds=param_binds,
    )
    result = job.result()
    counts = result.get_counts(circuit)

    # Convert counts to expectation values
    exp_vals = np.zeros((len(data), circuit.num_qubits))
    for key, val in counts.items():
        bits = np.array([int(b) for b in key[::-1]])  # reverse due to bitstring order
        prob = val / shots
        exp_vals += (2 * bits - 1) * prob
    return exp_vals / len(data)


__all__ = [
    "build_fraud_detection_circuit",
    "run_fraud_detection_circuit",
]
