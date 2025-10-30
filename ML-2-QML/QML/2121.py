"""Quantum estimator based on a 2‑qubit variational circuit with entanglement and multiple observables."""

from __future__ import annotations

from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import Estimator as StatevectorEstimator


def EstimatorQNN() -> QiskitEstimatorQNN:
    """
    Construct a 2‑qubit variational estimator.

    Returns
    -------
    qiskit_machine_learning.neural_networks.EstimatorQNN
    """

    # Define input and trainable parameters
    input_params = [Parameter("x1"), Parameter("x2")]
    weight_params = [Parameter(f"w{i}") for i in range(4)]

    # Variational circuit: H on both qubits, entangling CX, then RY/RZ parameterised
    qc = QuantumCircuit(2)
    qc.h([0, 1])
    qc.cx(0, 1)
    qc.ry(weight_params[0], 0)
    qc.rz(weight_params[1], 0)
    qc.ry(weight_params[2], 1)
    qc.rz(weight_params[3], 1)
    qc.cx(1, 0)  # additional entanglement

    # Observables: Y on each qubit and product Y⊗Y
    obs1 = SparsePauliOp.from_list([("Y", 1)])
    obs2 = SparsePauliOp.from_list([("I Y", 1)])
    obs3 = SparsePauliOp.from_list([("Y Y", 1)])
    observables = [obs1, obs2, obs3]

    estimator = StatevectorEstimator()
    estimator_qnn = QiskitEstimatorQNN(
        circuit=qc,
        observables=observables,
        input_params=input_params,
        weight_params=weight_params,
        estimator=estimator,
    )
    return estimator_qnn
