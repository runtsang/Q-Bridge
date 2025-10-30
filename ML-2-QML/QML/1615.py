"""
Quantum neural network for regression that uses a three‑qubit
variational circuit.  The circuit contains:

* Two parameter‑shared layers of single‑qubit rotations (Rx, Ry).
* An entangling CNOT ladder between all pairs.
* Separate weight parameters for each rotation.
* Observables: Pauli‑Z expectation on each qubit.

The network is wrapped by Qiskit Machine Learning's
`EstimatorQNN`, exposing a callable that can be trained with
gradient‑based optimisers.
"""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as _EstimatorQNN
from qiskit.primitives import StatevectorEstimator


def EstimatorQNN(
    input_dim: int = 2, num_qubits: int = 3, **kwargs
) -> _EstimatorQNN:
    """
    Build a parameterised quantum circuit and return a Qiskit
    Machine Learning EstimatorQNN instance.

    Parameters
    ----------
    input_dim : int, default=2
        Number of classical input features (used to set the number of input
        parameters on the first layer).
    num_qubits : int, default=3
        Number of qubits in the variational circuit.
    **kwargs
        Additional keyword arguments forwarded to the underlying
        `EstimatorQNN` constructor (e.g. `estimator`, `observables`).

    Returns
    -------
    qiskit_machine_learning.neural_networks.EstimatorQNN
        A Qiskit neural network ready for training.
    """
    # Create a list of parameters: one for each input feature per qubit
    input_params = []
    weight_params = []

    for qubit in range(num_qubits):
        # Input encoding: Ry with the classical feature
        for i in range(input_dim):
            p_in = Parameter(f"input_{qubit}_{i}")
            input_params.append(p_in)
        # Weight parameters for rotation gates
        p_wx = Parameter(f"weight_rx_{qubit}")
        p_wy = Parameter(f"weight_ry_{qubit}")
        weight_params.extend([p_wx, p_wy])

    qc = QuantumCircuit(num_qubits)

    # Layer 1: Encode inputs
    for qubit in range(num_qubits):
        for i in range(input_dim):
            idx = qubit * input_dim + i
            qc.ry(input_params[idx], qubit)

    # Layer 2: Entangling CNOT ladder
    for qubit in range(num_qubits - 1):
        qc.cx(qubit, qubit + 1)

    # Layer 3: Parameterised rotations
    for qubit in range(num_qubits):
        idx = 2 * qubit
        qc.rx(weight_params[idx], qubit)
        qc.ry(weight_params[idx + 1], qubit)

    # Observables: Pauli‑Z on each qubit
    observables = [
        SparsePauliOp.from_list([("Z" * num_qubits, 1)])
    ] * num_qubits

    # Default estimator if none provided
    estimator = kwargs.pop("estimator", StatevectorEstimator())

    return _EstimatorQNN(
        circuit=qc,
        observables=observables,
        input_params=input_params,
        weight_params=weight_params,
        estimator=estimator,
        **kwargs,
    )


__all__ = ["EstimatorQNN"]
