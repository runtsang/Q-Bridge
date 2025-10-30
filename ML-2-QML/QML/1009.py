"""Advanced variational quantum regressor with multi‑qubit entanglement."""
from __future__ import annotations

from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator


def EstimatorQNN() -> EstimatorQNN:
    """
    Construct a 3‑qubit variational circuit with entanglement and
    dual observables (Y and Z) for regression.
    """
    # Parameter definitions
    params = [
        Parameter(f"theta_{i}") for i in range(9)
    ]  # 3 qubits × 3 rotation parameters each

    qc = QuantumCircuit(3)

    # Layer 1: Parameterized rotations on each qubit
    for q in range(3):
        qc.ry(params[3 * q], q)
        qc.rz(params[3 * q + 1], q)
        qc.rx(params[3 * q + 2], q)

    # Entanglement layer
    qc.cx(0, 1)
    qc.cx(1, 2)

    # Second rotation layer (shared parameters for simplicity)
    for q in range(3):
        qc.ry(params[3 * q], q)
        qc.rz(params[3 * q + 1], q)
        qc.rx(params[3 * q + 2], q)

    # Observables: Y on qubit 0 and Z on qubit 2
    observable1 = SparsePauliOp.from_list([("Y" + "I" * 2, 1)])
    observable2 = SparsePauliOp.from_list([("I" * 2 + "Z", 1)])

    # Prepare the EstimatorQNN
    estimator = StatevectorEstimator()
    estimator_qnn = EstimatorQNN(
        circuit=qc,
        observables=[observable1, observable2],
        input_params=[params[0], params[3], params[6]],  # one per qubit
        weight_params=params[1:9],  # remaining parameters as weights
        estimator=estimator,
    )
    return estimator_qnn


__all__ = ["EstimatorQNN"]
