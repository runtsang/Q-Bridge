from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator

class EstimatorQNNGen227:
    """Quantum estimator mirroring the classical feed‑forward network.

    The circuit consists of an encoding layer followed by ``depth`` variational
    layers.  Each variational layer applies a single‑qubit rotation followed
    by a CZ coupling between neighbouring qubits.  The observable list is
    constructed to match the output dimension of the classical network.
    """

    def __init__(self,
                 num_qubits: int = 2,
                 depth: int = 2,
                 backend=None,
                 **kwargs) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.circuit, self.encoding, self.weights, self.observables = self._build_circuit()
        self.estimator = StatevectorEstimator(backend=backend, **kwargs)
        self.qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=self.observables,
            input_params=self.encoding,
            weight_params=self.weights,
            estimator=self.estimator,
        )

    def _build_circuit(self) -> tuple[QuantumCircuit, list[ParameterVector], list[ParameterVector], list[SparsePauliOp]]:
        encoding = ParameterVector("x", self.num_qubits)
        weights = ParameterVector("theta", self.num_qubits * self.depth)

        qc = QuantumCircuit(self.num_qubits)
        # Encoding: RX rotations
        for qubit, param in enumerate(encoding):
            qc.rx(param, qubit)

        # Variational layers
        idx = 0
        for _ in range(self.depth):
            for qubit in range(self.num_qubits):
                qc.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(self.num_qubits - 1):
                qc.cz(qubit, qubit + 1)

        # Observables: one Z per qubit
        observables = [SparsePauliOp("I" * i + "Z" + "I" * (self.num_qubits - i - 1))
                       for i in range(self.num_qubits)]
        return qc, list(encoding), list(weights), observables

    def evaluate(self,
                 inputs: list[float],
                 weight_vector: list[float]) -> list[float]:
        """Run the QNN with the provided classical inputs and variational weights.

        Parameters
        ----------
        inputs
            Classical data vector of length ``num_qubits``.
        weight_vector
            Flattened list of variational parameters matching ``weights``.
        """
        # Bind parameters
        bound_circuit = self.circuit.bind_parameters(
            {p: v for p, v in zip(self.encoding, inputs)}
            | {p: v for p, v in zip(self.weights, weight_vector)}
        )
        # Evaluate expectation values
        result = self.estimator(bound_circuit, self.observables)
        return result.values

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_qubits={self.num_qubits}, depth={self.depth})"
