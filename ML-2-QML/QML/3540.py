"""Quantum estimator that mirrors EstimatorQNN__gen166.

The implementation uses Qiskit to construct a parameter‑shaped
circuit.  Input parameters are encoded via Ry rotations; weight
parameters are encoded via Rx rotations.  An entangling layer
provides expressivity, and the expectation value of a Pauli‑Y
observable is used as the regression output.
"""

from __future__ import annotations

from qiskit.circuit import ParameterVector, QuantumCircuit, Parameter
from qiskit.quantum_info import Pauli
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator as StatevectorEstimator
from qiskit import Aer

# Circuit builder
def build_regression_circuit(
    n_qubits: int,
    input_params: ParameterVector,
    weight_params: ParameterVector,
) -> QuantumCircuit:
    """
    Construct a shallow variational circuit.

    The circuit applies:
        * Input encoding via Ry rotations
        * Entangling layer of CNOTs
        * Weight encoding via Rx rotations
    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    input_params : ParameterVector
        Parameters corresponding to input features.
    weight_params : ParameterVector
        Parameters corresponding to trainable weights.
    Returns
    -------
    QuantumCircuit
        The constructed circuit.
    """
    qc = QuantumCircuit(n_qubits)
    # Encode inputs
    for qubit, param in enumerate(input_params):
        qc.ry(param, qubit)
    # Entanglement
    for qubit in range(n_qubits - 1):
        qc.cx(qubit, qubit + 1)
    # Encode weights
    for qubit, param in enumerate(weight_params):
        qc.rx(param, qubit)
    return qc

# Observable
observable = Pauli('Y' * 1)  # single‑qubit Y observable

def get_estimator_qnn() -> EstimatorQNN:
    """
    Instantiate the Qiskit EstimatorQNN that reproduces the
    classical architecture’s parameter layout.

    Returns
    -------
    EstimatorQNN
        Quantum neural network estimator.
    """
    n_qubits = 1
    input_params = ParameterVector("x", size=1)
    weight_params = ParameterVector("w", size=1)

    qc = build_regression_circuit(n_qubits, input_params, weight_params)
    estimator = StatevectorEstimator(method="statevector", backend=Aer.get_backend("statevector_simulator"))

    qnn = EstimatorQNN(
        circuit=qc,
        observables=observable,
        input_params=input_params,
        weight_params=weight_params,
        estimator=estimator,
    )
    return qnn

__all__ = ["build_regression_circuit", "get_estimator_qnn"]
