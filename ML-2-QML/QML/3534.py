"""Quantum EstimatorQNN augmented with a two‑qubit self‑attention block.

The circuit applies a small self‑attention style unitary to two qubits,
then measures the Y Pauli operator on the first qubit to obtain a scalar
output. Parameters are split into an input parameter that controls the
first rotation and several weight parameters that drive the rest of the
circuit, allowing joint optimisation with a classical backend.
"""

from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QEstimatorQNN
from qiskit.primitives import StatevectorEstimator


def EstimatorQNN() -> QEstimatorQNN:
    """
    Construct a Qiskit EstimatorQNN that contains a two‑qubit
    self‑attention subcircuit followed by a measurement on the first qubit.

    Returns
    -------
    QEstimatorQNN
        Quantum neural network ready for training with a classical
        optimisation loop.
    """
    # Input parameter that will be mapped to the first rotation
    input_param = Parameter("input")

    # Rotation parameters for the two qubits (3 per qubit)
    rotation_params = [Parameter(f"rot{i}") for i in range(1, 6)]  # 5 parameters

    # Entanglement parameter between the two qubits
    entangle_params = [Parameter("ent1")]

    # Build the two‑qubit self‑attention circuit
    qc = QuantumCircuit(2)
    # Apply rotations to qubit 0
    qc.rx(input_param, 0)          # input‑dependent rotation
    qc.ry(rotation_params[0], 0)
    qc.rz(rotation_params[1], 0)
    # Apply rotations to qubit 1
    qc.rx(rotation_params[2], 1)
    qc.ry(rotation_params[3], 1)
    qc.rz(rotation_params[4], 1)
    # Entangle the qubits with a controlled‑R_x gate
    qc.crx(entangle_params[0], 0, 1)

    # Observable: Y on the first qubit
    observable = SparsePauliOp.from_list([("Y", 1)])

    # Quantum estimator primitive
    estimator = StatevectorEstimator()

    # Assemble the EstimatorQNN
    estimator_qnn = QEstimatorQNN(
        circuit=qc,
        observables=observable,
        input_params=[input_param],
        weight_params=rotation_params + entangle_params,
        estimator=estimator,
    )
    return estimator_qnn


__all__ = ["EstimatorQNN"]
