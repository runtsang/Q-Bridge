"""Quantum QCNN with entangling ansatz and multi‑observable readout."""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals

class QCNNEnhanced:
    """Quantum QCNN with a deep entangling ansatz and summed‑Z observable."""
    def __init__(self, num_qubits: int = 8, num_layers: int = 3) -> None:
        algorithm_globals.random_seed = 12345
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.feature_map = ZFeatureMap(num_qubits)
        self.ansatz = self._build_ansatz()
        self.observable = self._build_observable()
        self.estimator = Estimator()
        self.qnn = EstimatorQNN(
            circuit=self.ansatz.decompose(),
            observables=self.observable,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            estimator=self.estimator,
        )

    def _build_ansatz(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits, name="EntanglingAnsatz")
        params = ParameterVector("θ", length=self.num_qubits * self.num_layers * 3)
        idx = 0
        for _ in range(self.num_layers):
            # Parameterised rotations
            for q in range(self.num_qubits):
                qc.ry(params[idx], q); idx += 1
                qc.rz(params[idx], q); idx += 1
                qc.rx(params[idx], q); idx += 1
            # Ring entanglement
            for q in range(self.num_qubits):
                qc.cx(q, (q + 1) % self.num_qubits)
            qc.barrier()
        return qc

    def _build_observable(self) -> SparsePauliOp:
        # Sum of Z on all qubits
        paulis = [("Z" + "I" * (self.num_qubits - 1), 1.0)]
        for q in range(1, self.num_qubits):
            paulis.append(("I" * q + "Z" + "I" * (self.num_qubits - q - 1), 1.0))
        return SparsePauliOp.from_list(paulis)

    def __call__(self, *args, **kwargs):
        return self.qnn(*args, **kwargs)

def QCNNEnhanced() -> QCNNEnhanced:
    """Factory returning the configured QCNNEnhanced QNN."""
    return QCNNEnhanced()

__all__ = ["QCNNEnhanced"]
