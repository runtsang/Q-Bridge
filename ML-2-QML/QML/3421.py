"""Hybrid quantum estimator with a variational circuit.

This implementation builds upon the original EstimatorQNN example but
expands the circuit to 8 qubits.  Two input parameters encode the
feature vector via RY rotations, while eight variational parameters
parameterise a RZ layer on each qubit.  A CX ladder creates
entanglement, and the observable is a global Y operator.  The
estimator returns a single expectation value that can be used for
regression.
"""

from __future__ import annotations

from qiskit.circuit import Parameter
from qiskit import QuantumCircuit, Aer
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator as StatevectorEstimator


def HybridEstimator():
    """
    Construct a Qiskit EstimatorQNN that implements a hybrid quantum
    regression model.

    Returns
    -------
    EstimatorQNN
        A Qiskit neural network object ready for training.
    """
    # Input parameters (two features)
    input_params = [Parameter(f"input_{i}") for i in range(2)]
    # Variational parameters (one per qubit)
    weight_params = [Parameter(f"weight_{i}") for i in range(8)]

    # Build the circuit
    qc = QuantumCircuit(8)

    # Data encoding: RY rotations on the first two qubits
    qc.ry(input_params[0], 0)
    qc.ry(input_params[1], 1)

    # Variational layer: RZ on each qubit
    for i, w in enumerate(weight_params):
        qc.rz(w, i)

    # Entanglement ladder
    for i in range(7):
        qc.cx(i, i + 1)

    # Observable: global Y operator on all qubits
    observables = SparsePauliOp.from_list([("Y" * 8, 1)])

    # Estimator backend
    estimator = StatevectorEstimator(backend=Aer.get_backend("statevector_simulator"))

    # Wrap into EstimatorQNN
    estimator_qnn = EstimatorQNN(
        circuit=qc,
        observables=observables,
        input_params=input_params,
        weight_params=weight_params,
        estimator=estimator,
    )

    return estimator_qnn


__all__ = ["HybridEstimator"]
