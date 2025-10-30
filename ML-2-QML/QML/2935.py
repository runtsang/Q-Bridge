"""Quantum neural network that emulates the classical fraud‑layer structure.

The circuit consists of:
* An input layer that applies a Hadamard and a data‑dependent Ry rotation.
* Multiple variational layers, each containing a pair of Ry/Rz rotations on two qubits
  followed by CNOT entanglement.  The number of parameters per layer is four.
* A single‑qubit observable (Y on qubit 0) for expectation‑value estimation.

The public API matches the original EstimatorQNN example: a factory
``EstimatorQNN`` that returns a Qiskit ``EstimatorQNN`` instance.
"""

from __future__ import annotations

from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import StatevectorEstimator

def _layer_circuit(params: list[Parameter]) -> QuantumCircuit:
    """Create one variational layer with Ry/Rz rotations and CNOT entanglement."""
    qc = QuantumCircuit(2)
    qc.ry(params[0], 0)
    qc.rz(params[1], 0)
    qc.ry(params[2], 1)
    qc.rz(params[3], 1)
    qc.cx(0, 1)
    qc.cx(1, 0)
    return qc

def EstimatorQNN(
    n_layers: int = 3,
    input_params: list[Parameter] | None = None,
    weight_params: list[Parameter] | None = None,
) -> QiskitEstimatorQNN:
    """Return a Qiskit EstimatorQNN that mirrors the classical fraud‑layer design."""
    if input_params is None:
        input_params = [Parameter("x")]

    if weight_params is None:
        weight_params = [Parameter(f"w{layer}_{reg}") for layer in range(n_layers) for reg in range(4)]

    qc = QuantumCircuit(2)
    # Input encoding
    qc.h(0)
    qc.ry(input_params[0], 0)

    # Variational layers
    for i in range(n_layers):
        layer_params = weight_params[i * 4:(i + 1) * 4]
        qc += _layer_circuit(layer_params)

    # Measurement (required for StatevectorEstimator)
    qc.measure_all()

    observable = SparsePauliOp.from_list([("Y", 1)])
    estimator = StatevectorEstimator()
    estimator_qnn = QiskitEstimatorQNN(
        circuit=qc,
        observables=observable,
        input_params=input_params,
        weight_params=weight_params,
        estimator=estimator,
    )
    return estimator_qnn

__all__ = ["EstimatorQNN"]
