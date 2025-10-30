from __future__ import annotations

from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN

class EstimatorQNN(QiskitEstimatorQNN):
    """
    Quantum neural network based on a variational RealAmplitudes ansatz.
    The circuit applies input rotations RY for each qubit followed by the
    ansatz. The observable is the sum of Pauli‑Z operators on all qubits.
    """
    def __init__(self, num_qubits: int = 3) -> None:
        circuit, input_params, weight_params = self._build_circuit(num_qubits)
        observable = SparsePauliOp.from_list([("Z" * num_qubits, 1)])
        estimator = StatevectorEstimator()
        super().__init__(
            circuit=circuit,
            observables=observable,
            input_params=input_params,
            weight_params=weight_params,
            estimator=estimator,
        )

    @staticmethod
    def _build_circuit(num_qubits: int):
        """Create a circuit with RY input rotations and a RealAmplitudes ansatz."""
        input_params = [Parameter(f"input_{i}") for i in range(num_qubits)]
        ansatz = RealAmplitudes(num_qubits, reps=2)
        qc = QuantumCircuit(num_qubits)
        for i, p in enumerate(input_params):
            qc.ry(p, i)
        qc.compose(ansatz, range(num_qubits), inplace=True)
        return qc, input_params, ansatz.parameters
