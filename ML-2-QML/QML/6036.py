import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

class HybridSelfAttentionClassifierQML:
    """
    Quantum implementation of a hybrid self‑attention + classifier.
    The first block is a variational attention circuit that encodes
    the input features into a quantum state via rotations and
    CZ‑style entanglement.  The second block is a layered ansatz
    similar to the reference quantum classifier.  The two blocks
    are concatenated into a single circuit that is executed on a
    backend and returns measurement statistics.
    """

    def __init__(
        self,
        num_qubits: int,
        attention_depth: int,
        classifier_depth: int,
        backend=None,
        shots: int = 1024,
    ):
        self.num_qubits = num_qubits
        self.attention_depth = attention_depth
        self.classifier_depth = classifier_depth
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots

        # Parameter vectors for the two blocks
        self.attention_params = ParameterVector("a", num_qubits * 3 * attention_depth)
        self.classifier_params = ParameterVector("c", num_qubits * classifier_depth)

        # Build the full circuit
        self.circuit = self._build_circuit()

    def _build_attention(self, qc: QuantumCircuit, params: ParameterVector):
        idx = 0
        for _ in range(self.attention_depth):
            for qubit in range(self.num_qubits):
                qc.rx(params[idx], qubit); idx += 1
                qc.ry(params[idx], qubit); idx += 1
                qc.rz(params[idx], qubit); idx += 1
            for qubit in range(self.num_qubits - 1):
                qc.cz(qubit, qubit + 1)

    def _build_classifier(self, qc: QuantumCircuit, params: ParameterVector):
        idx = 0
        for _ in range(self.classifier_depth):
            for qubit in range(self.num_qubits):
                qc.ry(params[idx], qubit); idx += 1
            for qubit in range(self.num_qubits - 1):
                qc.cz(qubit, qubit + 1)

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits, self.num_qubits)
        self._build_attention(qc, self.attention_params)
        self._build_classifier(qc, self.classifier_params)
        qc.measure_all()
        return qc

    def run(
        self,
        attention_params: np.ndarray,
        classifier_params: np.ndarray,
    ) -> dict:
        """
        Execute the hybrid circuit with supplied parameters.
        Parameters are flattened arrays matching the internal
        parameter vectors.
        """
        bound_circuit = self.circuit.bind_parameters(
            {
                **{p: val for p, val in zip(self.attention_params, attention_params)},
                **{p: val for p, val in zip(self.classifier_params, classifier_params)},
            }
        )
        job = qiskit.execute(bound_circuit, self.backend, shots=self.shots)
        return job.result().get_counts(bound_circuit)

    @property
    def observables(self) -> list[SparsePauliOp]:
        """
        Return a list of Z observables on each qubit – analogous to
        the classical classifier output neurons.
        """
        return [
            SparsePauliOp("I" * i + "Z" + "I" * (self.num_qubits - i - 1))
            for i in range(self.num_qubits)
        ]

__all__ = ["HybridSelfAttentionClassifierQML"]
