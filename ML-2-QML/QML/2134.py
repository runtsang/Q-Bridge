"""
Hybrid quantum classifier circuit factory and measurement routine.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.opflow import StateFn, ExpectationFactory, PauliExpectation, SummedOp
from qiskit.providers import Backend
from qiskit.providers.aer import AerSimulator
from typing import Tuple, Iterable, List


def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
    """
    Construct a variational ansatz with an encoding layer, repeated entanglement blocks,
    and a list of observables for a multi‑class readout.

    Parameters
    ----------
    num_qubits : int
        Number of qubits representing the feature vector.
    depth : int
        Number of variational layers.

    Returns
    -------
    circuit : QuantumCircuit
        The variational circuit with symbolic parameters.
    encodings : list of ParameterVector
        Encoding parameters (one per qubit).
    weights : list of ParameterVector
        Variational parameters (one per qubit per depth).
    observables : list of SparsePauliOp
        Measurement operators that encode a class label.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = [ParameterVector(f"theta_{d}", num_qubits) for d in range(depth)]

    circuit = QuantumCircuit(num_qubits)
    # Data encoding: RX rotations
    for qubit in range(num_qubits):
        circuit.rx(encoding[qubit], qubit)

    # Variational layers with tunable entanglement
    for d, theta in enumerate(weights):
        for qubit in range(num_qubits):
            circuit.ry(theta[qubit], qubit)
        if d < depth - 1:
            for qubit in range(num_qubits - 1):
                circuit.cz(qubit, qubit + 1)

    # Readout observables: one Pauli-Z per qubit
    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]

    return circuit, [encoding], weights, observables


def run_quantum_circuit(
    circuit: QuantumCircuit,
    enc_params: List[float],
    var_params: List[float],
    backend: Backend,
    shots: int = 1024,
) -> np.ndarray:
    """
    Execute the circuit on the given backend and return the expectation values for the observables.

    Parameters
    ----------
    circuit : QuantumCircuit
        The symbolic circuit returned by :func:`build_classifier_circuit`.
    enc_params : list[float]
        Numerical values for the encoding parameters.
    var_params : list[float]
        Numerical values for the variational parameters.
    backend : Backend
        Qiskit backend (AerSimulator or a real device).
    shots : int
        Number of shots for a stochastic simulation.

    Returns
    -------
    np.ndarray
        Array of expectation values, shape (num_qubits,).
    """
    # Bind parameters
    param_dict = dict(zip(circuit.parameters, enc_params + var_params))
    bound_circuit = circuit.bind_parameters(param_dict)

    # Prepare expectation evaluation
    state = StateFn(bound_circuit, backend=backend)
    exp = ExpectationFactory.build(operator=SummedOp(circuit.measure_all()), expectation=PauliExpectation())
    results = exp.convert(state, backend=backend)

    return np.array(results.data)


__all__ = ["build_classifier_circuit", "run_quantum_circuit"]
