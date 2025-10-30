"""Hybrid quantum‑classical estimator that embeds a CNN feature extractor into a variational circuit.

The QML module builds a quantum circuit that accepts features from a classical CNN,
encodes them via Ry gates, then applies a depth‑controlled variational layer
composed of RX, RY, RZ, and controlled‑RZ rotations.  The expectation value of a
multi‑qubit Y observable is used as the output.  The circuit is wrapped by
qiskit's EstimatorQNN neural network wrapper for seamless integration with
PyTorch training loops.
"""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import StatevectorEstimator
import torch


def EstimatorQNN(
    num_qubits: int = 1,
    depth: int = 3,
    use_cnn: bool = True,
) -> QiskitEstimatorQNN:
    """
    Build a hybrid quantum‑classical estimator.

    Parameters
    ----------
    num_qubits: int
        Number of qubits to use for the variational circuit.
    depth: int
        Number of variational layers.
    use_cnn: bool
        If True, prepend a classical 2‑D CNN feature extractor before the quantum circuit.
    """
    # --- Classical feature extractor (optional) --------------------------------
    if use_cnn:
        # A lightweight CNN that outputs a flattened feature vector.
        # This part is purely classical and can be executed on a CPU.
        cnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )
    else:
        cnn = None

    # --- Quantum circuit -------------------------------------------------------
    input_params = [Parameter(f"input_{i}") for i in range(num_qubits)]
    weight_params = [
        Parameter(f"theta_{d}_{i}")
        for d in range(depth)
        for i in range(num_qubits * 3)  # RX, RY, RZ per qubit
    ]

    qc = QuantumCircuit(num_qubits)

    # Encode classical features into the circuit
    for i, p in enumerate(input_params):
        qc.ry(p, i)

    # Variational layers
    for d in range(depth):
        base = d * num_qubits * 3
        for q in range(num_qubits):
            qc.rx(weight_params[base + q], q)
            qc.ry(weight_params[base + num_qubits + q], q)
            qc.rz(weight_params[base + 2 * num_qubits + q], q)
        # Entanglement: a simple linear chain of CX gates
        for q in range(num_qubits - 1):
            qc.cx(q, q + 1)

    # Observable: multi‑qubit Y operator
    observable = SparsePauliOp.from_list([("Y" * num_qubits, 1)])

    # Estimator primitive
    estimator = StatevectorEstimator()

    # Wrap into Qiskit EstimatorQNN
    estimator_qnn = QiskitEstimatorQNN(
        circuit=qc,
        observables=observable,
        input_params=input_params,
        weight_params=weight_params,
        estimator=estimator,
    )

    # Attach the classical CNN as a preprocessing step if requested
    if cnn is not None:
        # Store the CNN as an attribute for external use
        estimator_qnn.cnn = cnn

    return estimator_qnn


__all__ = ["EstimatorQNN"]
