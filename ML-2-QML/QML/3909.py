"""
Hybrid quantum estimator that mirrors the classical EstimatorQNN.
It builds a variational circuit with parameter‑vector encoding,
depth‑controlled layers, and a set of observables that can be treated
as regression outputs or logits for binary classification.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import Estimator

def build_quantum_circuit(
    num_qubits: int,
    depth: int,
    num_outputs: int,
) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
    """
    Construct a layered ansatz with explicit encoding and variational parameters.

    Parameters
    ----------
    num_qubits : int
        Number of qubits / input features.
    depth : int
        Number of variational layers.
    num_outputs : int
        Number of observable expectation values to return (e.g. 1 for regression,
        2 for binary classification).

    Returns
    -------
    tuple
        (circuit, encoding_params, weight_params, observables)
    """
    # Encoding: one RX per feature
    encoding = ParameterVector("x", num_qubits)
    # Variational angles
    weights = ParameterVector("theta", num_qubits * depth)

    qc = QuantumCircuit(num_qubits)

    # Feature encoding
    for i in range(num_qubits):
        qc.rx(encoding[i], i)

    # Variational layers
    w_idx = 0
    for _ in range(depth):
        for i in range(num_qubits):
            qc.ry(weights[w_idx], i)
            w_idx += 1
        # Entangling layer
        for i in range(num_qubits - 1):
            qc.cz(i, i + 1)

    # Observables
    observables: List[SparsePauliOp] = []
    for i in range(num_outputs):
        # Expectation of Z on qubit i (or the first qubit if only one output)
        pauli = ("Z" * i + "I" * (num_qubits - i - 1))
        observables.append(SparsePauliOp.from_list([(pauli, 1)]))

    return qc, [encoding], [weights], observables


def EstimatorQNN(
    num_qubits: int,
    depth: int,
    num_outputs: int,
    backend_name: str = "statevector_simulator",
) -> QiskitEstimatorQNN:
    """
    Wrap the quantum circuit in a Qiskit Machine Learning EstimatorQNN.

    Parameters
    ----------
    num_qubits : int
        Number of qubits / input features.
    depth : int
        Number of variational layers.
    num_outputs : int
        Number of observables (output dimensions).
    backend_name : str, default "statevector_simulator"
        Quantum backend to evaluate the circuit.

    Returns
    -------
    qiskit_machine_learning.neural_networks.EstimatorQNN
        A hybrid estimator ready for training with the Qiskit Estimator.
    """
    circuit, enc, wts, obs = build_quantum_circuit(num_qubits, depth, num_outputs)

    estimator = Estimator(backend=backend_name)

    estimator_qnn = QiskitEstimatorQNN(
        circuit=circuit,
        observables=obs,
        input_params=enc,
        weight_params=wts,
        estimator=estimator,
    )
    return estimator_qnn


__all__ = ["build_quantum_circuit", "EstimatorQNN"]
