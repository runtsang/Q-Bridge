"""Quantum neural network for regression with a scalable variational circuit.

build_regressor_circuit constructs a parameterised ansatz with data encoding by RX gates, a configurable depth of variational layers, and entangling CX gates.  The observables are Pauli‑Z on each qubit, yielding a feature vector that is interpreted as a regression target by a linear head in EstimatorQNN."""
from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator


def build_regressor_circuit(num_qubits: int, quantum_depth: int) -> tuple[QuantumCircuit, list[ParameterVector], list[ParameterVector], list[SparsePauliOp]]:
    """
    Build a variational ansatz suitable for regression.

    Parameters
    ----------
    num_qubits:
        Number of qubits (also the dimensionality of the input data).
    quantum_depth:
        Number of layers in the variational circuit.
    """
    # Encoding of the input data
    encoding = ParameterVector("x", num_qubits)
    # Variational parameters
    weights = ParameterVector("theta", num_qubits * quantum_depth)

    qc = QuantumCircuit(num_qubits)
    # Data encoding: RX gates
    for q, param in enumerate(encoding):
        qc.rx(param, q)

    # Ansatz layers
    idx = 0
    for _ in range(quantum_depth):
        for q in range(num_qubits):
            qc.ry(weights[idx], q)
            idx += 1
        # Entangling layer
        for q in range(num_qubits - 1):
            qc.cx(q, q + 1)

    # Observables: Z on each qubit
    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]

    return qc, list(encoding), list(weights), observables


def EstimatorQNNGen219(num_qubits: int, quantum_depth: int = 1) -> EstimatorQNN:
    """
    Return a quantum neural network ready for regression tasks.

    The network is built on top of Qiskit's EstimatorQNN wrapper,
    using the circuit from `build_regressor_circuit`.
    """
    qc, enc_params, w_params, obsvs = build_regressor_circuit(num_qubits, quantum_depth)

    # Estimator primitive
    estimator = StatevectorEstimator()

    # The head of the EstimatorQNN is a linear regression, i.e. the output
    # vector of expectation values is fed into a linear map to a single
    # scalar.  This is achieved by setting `input_params` to all encoding
    # parameters and `weight_params` to all variational parameters.
    return EstimatorQNN(
        circuit=qc,
        observables=obsvs,
        input_params=enc_params,
        weight_params=w_params,
        estimator=estimator,
    )
