"""Quantum neural network with 2 qubits, entanglement, and multiple observables."""

from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import Estimator as StatevectorEstimator

def EstimatorQNN() -> QiskitEstimatorQNN:
    """
    Construct a 2‑qubit variational circuit with entanglement and
    return a Qiskit EstimatorQNN object.
    """
    # Parameter definitions
    input_params = [Parameter(f"inp_{i}") for i in range(2)]
    weight_params = [Parameter(f"w_{i}") for i in range(2)]

    # Variational circuit
    qc = QuantumCircuit(2)
    # Encoding layer
    qc.ry(input_params[0], 0)
    qc.rz(input_params[1], 1)
    # Entanglement
    qc.cx(0, 1)
    # Variational layer
    qc.ry(weight_params[0], 0)
    qc.rz(weight_params[1], 1)
    qc.cx(1, 0)

    # Observables: Z on each qubit
    obs_z0 = SparsePauliOp.from_list([("ZI", 1)])  # Z on qubit 0
    obs_z1 = SparsePauliOp.from_list([("IZ", 1)])  # Z on qubit 1

    # State‑vector estimator
    estimator = StatevectorEstimator()

    return QiskitEstimatorQNN(
        circuit=qc,
        observables=[obs_z0, obs_z1],
        input_params=input_params,
        weight_params=weight_params,
        estimator=estimator,
    )

__all__ = ["EstimatorQNN"]
