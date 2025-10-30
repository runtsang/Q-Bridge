"""Variational quantum neural network for regression.

The circuit uses two qubits and a repeatable entangling block.
Input data is encoded via RX rotations; trainable weights are
parameterised by RZ rotations.  The network outputs the expectation
value of a Pauli-Y observable on each qubit, concatenated to form
a scalar prediction.
"""

from qiskit import QuantumCircuit, Aer
from qiskit.circuit import Parameter
from qiskit.quantum_info import Pauli
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import Estimator as QiskitEstimator
from typing import Tuple, List

def EstimatorQNN(
    n_qubits: int = 2,
    n_layers: int = 2,
    backend_name: str = "statevector_simulator",
) -> QiskitEstimatorQNN:
    """
    Build a variational quantum neural network.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit.
    n_layers : int
        Number of repeatable parameterised layers.
    backend_name : str
        Aer backend to use for statevector estimation.
    """
    # Parameters
    input_params = [Parameter(f"inp_{i}") for i in range(n_qubits)]
    weight_params = [Parameter(f"w_{i}") for i in range(n_qubits * n_layers * 2)]

    qc = QuantumCircuit(n_qubits)

    # Data re-uploading circuit
    for layer in range(n_layers):
        for q in range(n_qubits):
            qc.rx(input_params[q], q)
            qc.rz(weight_params[layer * n_qubits * 2 + q], q)
        # Entangling block
        for q in range(n_qubits - 1):
            qc.cz(q, q + 1)
        for q in range(n_qubits - 1):
            qc.cz(q + 1, q)

    # Observables: Pauli-Y on each qubit
    observables: List[Pauli] = [
        Pauli("Y" + "I" * (n_qubits - 1 - i) + "I" * i) for i in range(n_qubits)
    ]

    # Estimator
    backend = Aer.get_backend(backend_name)
    estimator = QiskitEstimator(backend=backend)

    return QiskitEstimatorQNN(
        circuit=qc,
        observables=observables,
        input_params=input_params,
        weight_params=weight_params,
        estimator=estimator,
    )

__all__ = ["EstimatorQNN"]
