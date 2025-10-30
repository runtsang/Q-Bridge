"""Variational quantum neural network with entanglement and multiple observables.

The circuit now uses a two‑qubit entangling block, parameterized rotations,
and an observable that is a linear combination of Pauli‑Z operators.
"""

from qiskit.circuit import ParameterVector
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import Estimator as StatevectorEstimator
from typing import List

class EstimatorQNNExtended:
    """Wrapper around a Qiskit variational circuit that can be used as a QML model.

    Parameters
    ----------
    num_qubits : int, default 2
        Number of qubits in the circuit.
    depth : int, default 2
        Number of variational layers.
    """

    def __init__(self, num_qubits: int = 2, depth: int = 2) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.input_params = ParameterVector("x", num_qubits)
        self.weight_params = ParameterVector("w", num_qubits * depth * 3)

        self.circuit = self._build_circuit()
        self.observables = self._build_observables()
        self.estimator = StatevectorEstimator()
        self.estimator_qnn = QiskitEstimatorQNN(
            circuit=self.circuit,
            observables=self.observables,
            input_params=self.input_params,
            weight_params=self.weight_params,
            estimator=self.estimator,
        )

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)
        # Input encoding
        for i, param in enumerate(self.input_params):
            qc.ry(param, i)
        # Variational layers
        idx = 0
        for _ in range(self.depth):
            for q in range(self.num_qubits):
                qc.rx(self.weight_params[idx], q); idx += 1
                qc.ry(self.weight_params[idx], q); idx += 1
                qc.rz(self.weight_params[idx], q); idx += 1
            # Entangling block
            for q in range(self.num_qubits - 1):
                qc.cx(q, q + 1)
        return qc

    def _build_observables(self) -> List[SparsePauliOp]:
        # Observable is a weighted sum of Z on each qubit
        obs_list = []
        for i in range(self.num_qubits):
            coeff = 1.0
            pauli = "Z" * self.num_qubits
            pauli = pauli[:i] + "Z" + pauli[i + 1 :]
            obs_list.append((pauli, coeff))
        return [SparsePauliOp.from_list(obs_list)]

    def __call__(self, inputs: List[List[float]]) -> List[float]:
        """Evaluate the circuit for a batch of inputs.

        Parameters
        ----------
        inputs : list[list[float]]
            Batch of input feature vectors of length ``num_qubits``.
        """
        return self.estimator_qnn.predict(inputs)

def EstimatorQNN() -> EstimatorQNNExtended:
    """Convenience factory that returns the default configuration."""
    return EstimatorQNNExtended()

__all__ = ["EstimatorQNNExtended", "EstimatorQNN"]
