from qiskit.circuit import ParameterVector, Parameter
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator as StatevectorEstimator
import numpy as np


def _build_quantum_quanvolution_circuit(input_params: ParameterVector,
                                        weight_params: ParameterVector) -> QuantumCircuit:
    """
    Builds a 4‑qubit variational circuit that encodes a 2×2 image patch.
    Each input pixel is encoded via an Ry rotation; the remaining
    parameters form a shallow trainable variational layer.
    """
    qc = QuantumCircuit(4)
    # Input encoding
    for qubit, inp in enumerate(input_params):
        qc.ry(inp, qubit)

    # Trainable variational layers (simple Rz rotations)
    for qubit, w in enumerate(weight_params):
        qc.rz(w, qubit % 4)

    # Entangling gates to mix information across qubits
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)
    return qc


def EstimatorQNN():
    """
    Returns a qiskit_estimator neural network that implements the quantum
    quanvolution filter followed by a linear measurement.  It exposes the same
    EstimatorQNN API as the classical implementation, enabling direct
    comparison in mixed‑precision experiments.
    """
    # 4 input parameters – one per pixel in the 2×2 patch
    input_params = ParameterVector('input', 4)
    # 8 trainable weight parameters – one per qubit per layer
    weight_params = ParameterVector('weight', 8)

    # Assemble the circuit
    circuit = _build_quantum_quanvolution_circuit(input_params, weight_params)

    # Observable used for measurement – here a simple Pauli‑Z operator on all qubits
    observable = SparsePauliOp.from_list([('Z' * 4, 1.0)])

    # Quantum estimator backend
    estimator = StatevectorEstimator()

    # Construct the EstimatorQNN neural network
    qnn = EstimatorQNN(circuit=circuit,
                       observables=observable,
                       input_params=list(input_params),
                       weight_params=list(weight_params),
                       estimator=estimator)
    return qnn


__all__ = ["EstimatorQNN"]
