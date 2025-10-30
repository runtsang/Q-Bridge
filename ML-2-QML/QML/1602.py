"""Enhanced quantum sampler network using a deeper parameterized ansatz."""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN as QiskitSamplerQNN
from qiskit.primitives import Sampler as QiskitSampler


def SamplerQNN(
    num_qubits: int = 4,
    num_layers: int = 2,
    backend: str = "qasm_simulator",
) -> QiskitSamplerQNN:
    """
    Construct a parameterized quantum circuit for sampling with depth control.
    Parameters
    ----------
    num_qubits : int
        Number of qubits in the ansatz.
    num_layers : int
        Number of entangling layers.
    backend : str
        Backend to use for sampling ("qasm_simulator", "statevector", etc.).
    Returns
    -------
    QiskitSamplerQNN
        A Qiskit SamplerQNN instance ready for use.
    """
    # Input and weight parameters
    inputs = ParameterVector("input", num_qubits)
    weights = ParameterVector("weight", num_qubits * num_layers * 2)

    qc = QuantumCircuit(num_qubits)

    param_idx = 0
    # Initial rotation
    for qubit in range(num_qubits):
        qc.ry(inputs[qubit], qubit)

    # Entangling layers
    for _ in range(num_layers):
        # Rotation weights
        for qubit in range(num_qubits):
            qc.ry(weights[param_idx], qubit)
            param_idx += 1
        # Entangling CX chain
        for q in range(num_qubits - 1):
            qc.cx(q, q + 1)
        for q in reversed(range(num_qubits - 1)):
            qc.cx(q, q + 1)
        # Second set of rotation weights
        for qubit in range(num_qubits):
            qc.ry(weights[param_idx], qubit)
            param_idx += 1

    # Choose sampler backend
    sampler = QiskitSampler(method=backend)

    sampler_qnn = QiskitSamplerQNN(
        circuit=qc,
        input_params=inputs,
        weight_params=weights,
        sampler=sampler,
    )
    return sampler_qnn


__all__ = ["SamplerQNN"]
