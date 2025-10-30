"""Quantum counterpart of EstimatorQNN using a quanvolution filter and
Qiskit’s EstimatorQNN for hybrid training.

The circuit first applies a parameterised quanvolution on the input
image, then a shallow variational circuit whose expectation value
serves as the output.  The module is ready for use with Qiskit
Machine Learning’s primitives."""
from __future__ import annotations

import numpy as np
import qiskit
from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import StatevectorEstimator

def QuanvCircuit(kernel_size: int = 2,
                 backend: qiskit.providers.Backend = None,
                 shots: int = 100,
                 threshold: float = 127.0) -> QuantumCircuit:
    """
    Build a quanvolution circuit that encodes a 2‑D patch into qubit rotations.
    """
    if backend is None:
        backend = qiskit.Aer.get_backend("qasm_simulator")

    n_qubits = kernel_size ** 2
    circuit = QuantumCircuit(n_qubits)

    # Parameterised RX gates to encode pixel values
    theta = [Parameter(f"theta{idx}") for idx in range(n_qubits)]
    for idx in range(n_qubits):
        circuit.rx(theta[idx], idx)

    circuit.barrier()

    # Add a shallow random circuit to mix the qubits
    circuit += qiskit.circuit.random.random_circuit(n_qubits, depth=2)

    # Measure all qubits to obtain a probability distribution
    circuit.measure_all()
    circuit.backend = backend
    circuit.shots = shots
    circuit.threshold = threshold
    return circuit


def EstimatorQNN(kernel_size: int = 2,
                 backend: qiskit.providers.Backend = None,
                 shots: int = 100,
                 threshold: float = 127.0) -> QiskitEstimatorQNN:
    """
    Construct a hybrid quantum estimator that uses a quanvolution
    circuit as the input layer and a variational circuit for regression.
    """
    if backend is None:
        backend = qiskit.Aer.get_backend("qasm_simulator")

    # Quanvolution circuit
    quanv = QuanvCircuit(kernel_size=kernel_size,
                         backend=backend,
                         shots=shots,
                         threshold=threshold)

    # Parameterised variational circuit (simple 1‑qubit rotation)
    var_circ = QuantumCircuit(1)
    w = Parameter("weight1")
    var_circ.rx(w, 0)

    # Observables: expectation of Pauli Y on the variational qubit
    from qiskit.quantum_info import SparsePauliOp
    observable = SparsePauliOp.from_list([("Y", 1)])

    # Wrap into Qiskit EstimatorQNN
    estimator = StatevectorEstimator()
    estimator_qnn = QiskitEstimatorQNN(
        circuit=var_circ,
        observables=observable,
        input_params=[quanv.parameters[0]],          # use first pixel as input
        weight_params=[w],
        estimator=estimator,
    )
    return estimator_qnn


__all__ = ["EstimatorQNN", "QuanvCircuit"]
