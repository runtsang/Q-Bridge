from dataclasses import dataclass
from typing import List
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

@dataclass
class QuantumEstimator:
    """Container for all components required by the Qiskit EstimatorQNN."""
    circuit: QuantumCircuit
    input_params: List[Parameter]
    weight_params: List[Parameter]
    observable: SparsePauliOp
    estimator: StatevectorEstimator

def build_quantum_estimator() -> QuantumEstimator:
    """
    Builds a 4‑qubit variational circuit that accepts a 4‑dimensional input
    encoding and a single trainable weight.  The observable is a global Y
    operator, enabling a non‑trivial expectation value.
    """
    input_params = [Parameter(f"input_{i}") for i in range(4)]
    weight_params = [Parameter("weight_0")]

    qc = QuantumCircuit(4)
    # Input encoding: Ry rotations
    for i, p in enumerate(input_params):
        qc.ry(p, i)

    # Simple variational layer: global Z rotations + entangling CNOT chain
    for i in range(4):
        qc.rz(weight_params[0], i)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)

    observable = SparsePauliOp.from_list([("Y" * qc.num_qubits, 1)])
    estimator = StatevectorEstimator()

    return QuantumEstimator(
        circuit=qc,
        input_params=input_params,
        weight_params=weight_params,
        observable=observable,
        estimator=estimator,
    )

def get_estimator_qnn() -> EstimatorQNN:
    """
    Exposes a ready‑to‑use Qiskit EstimatorQNN that can be passed to the
    CombinedEstimatorQNN class.
    """
    qe = build_quantum_estimator()
    return EstimatorQNN(
        circuit=qe.circuit,
        observables=qe.observable,
        input_params=qe.input_params,
        weight_params=qe.weight_params,
        estimator=qe.estimator,
    )

__all__ = ["QuantumEstimator", "build_quantum_estimator", "get_estimator_qnn"]
