"""Quantum estimator that uses a variational circuit and the classical parameters produced by the HybridNN.

The circuit is constructed similarly to QuantumClassifierModel's build_classifier_circuit,
but it accepts external weight parameters via a ParameterVector that can be set by the
classical network. The EstimatorQNN class wraps Qiskit’s EstimatorQNN and exposes a
`set_q_params` method to inject the classical parameters.
"""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import StatevectorEstimator

def EstimatorQNN(num_qubits: int = 1, depth: int = 2) -> QiskitEstimatorQNN:
    """Return a quantum estimator that can be driven by classical parameters.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the ansatz.
    depth : int
        Number of variational layers.

    Returns
    -------
    QiskitEstimatorQNN
        A quantum estimator that accepts weight parameters via `set_q_params`.
    """
    # Build circuit with parameter vectors for inputs and weights
    input_params = ParameterVector("x", num_qubits)
    weight_params = ParameterVector("theta", num_qubits * depth)

    qc = QuantumCircuit(num_qubits)
    # Encoding
    for qubit, param in enumerate(input_params):
        qc.rx(param, qubit)

    # Variational layers
    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            qc.ry(weight_params[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            qc.cz(qubit, qubit + 1)

    # Observables: Z on each qubit
    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]

    # Primitive
    estimator = StatevectorEstimator()

    # Create the EstimatorQNN
    estimator_qnn = QiskitEstimatorQNN(
        circuit=qc,
        observables=observables,
        input_params=list(input_params),
        weight_params=list(weight_params),
        estimator=estimator,
    )

    # Expose a helper to set the weight parameters from a classical vector
    def set_q_params(self, q_params: list[float]) -> None:
        """Inject classical parameters into the quantum circuit."""
        if len(q_params)!= len(weight_params):
            raise ValueError(f"Expected {len(weight_params)} weight parameters, got {len(q_params)}.")
        param_bindings = {param: val for param, val in zip(weight_params, q_params)}
        self.circuit.assign_parameters(param_bindings, inplace=True)

    # Attach the helper to the estimator instance
    setattr(estimator_qnn, "set_q_params", set_q_params.__get__(estimator_qnn))

    return estimator_qnn
