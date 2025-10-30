import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
import numpy as np
from typing import Iterable, Tuple, List

def build_classifier_circuit(num_qubits: int,
                             depth: int) -> Tuple[QuantumCircuit,
                                                   List[ParameterVector],
                                                   List[ParameterVector],
                                                   List[SparsePauliOp]]:
    """
    Construct a data‑uploading variational circuit that mirrors the
    classical build function.

    Returns
    -------
    circuit : QuantumCircuit
        The variational ansatz with explicit RX encoding and
        parameterized Ry rotations.
    encoding : List[ParameterVector]
        List containing the encoding parameters (one per qubit).
    weights : List[ParameterVector]
        Trainable weight parameters for the Ry rotations.
    observables : List[SparsePauliOp]
        Pauli‑Z operators used for measurement.
    """
    # Encoding: one RX per qubit
    encoding = ParameterVector("x", num_qubits)

    # Variational parameters
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    # Apply data encoding
    for qubit, param in enumerate(encoding):
        circuit.rx(param, qubit)

    # Layered ansatz
    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    # Measurement observables
    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
                   for i in range(num_qubits)]
    return circuit, [encoding], [weights], observables

__all__ = ["build_classifier_circuit"]

class UnifiedQuantumClassifier:
    """
    Quantum wrapper that prepares a state, applies the variational circuit,
    and measures in the computational basis using the Pauli‑Z operators.
    It expects the same encoding and weight vectors that the
    ``build_classifier_circuit`` function returns.

    Parameters
    ----------
    num_qubits : int
        Number of qubits / features.
    depth : int
        Depth of the variational ansatz.
    backend : str, default='qasm_simulator'
        Qiskit backend name.
    """

    def __init__(self,
                 num_qubits: int,
                 depth: int = 2,
                 backend: str = "qasm_simulator"):
        self.num_qubits = num_qubits
        self.depth = depth
        self.backend = backend
        self.circuit, self.encoding, self.weights, self.observables = build_classifier_circuit(num_qubits, depth)
        self.backend_obj = qiskit.execute([], backend=backend, shots=1).backend

    def forward(self,
                data: np.ndarray,
                weight_values: np.ndarray) -> np.ndarray:
        """
        Execute the circuit for each data point in `data` and return
        expectation values of the Z observables.

        Parameters
        ----------
        data : np.ndarray, shape (batch, num_qubits)
            Classical feature vectors, each component in [0, 2π] for RX encoding.
        weight_values : np.ndarray, shape (num_qubits * depth,)
            Trainable parameters for the variational Ry gates.

        Returns
        -------
        output : np.ndarray, shape (batch, num_qubits)
            Expectation values of the Pauli‑Z operators.
        """
        batch_size = data.shape[0]
        results = np.zeros((batch_size, self.num_qubits))
        for i, x in enumerate(data):
            bound_circ = self.circuit.bind_parameters(
                {p: val for p, val in zip(self.encoding[0], x)}
            )
            bound_circ = bound_circ.bind_parameters(
                {p: val for p, val in zip(self.weights[0], weight_values)}
            )
            job = qiskit.execute(bound_circ, backend=self.backend_obj, shots=1024)
            counts = job.result().get_counts()
            # Convert counts to expectation values
            exp_vals = self._counts_to_expectation(counts)
            results[i] = exp_vals
        return results

    @staticmethod
    def _counts_to_expectation(counts: dict) -> np.ndarray:
        """
        Convert shot counts to expectation values of Z operators.
        """
        exp = np.zeros(len(counts))
        total = sum(counts.values())
        for state, n in counts.items():
            # state string is like '0101' where 0->+1, 1->-1
            for idx, bit in enumerate(reversed(state)):
                exp[idx] += ((-1) ** int(bit)) * n
        return exp / total

    def get_parameter_vector(self) -> np.ndarray:
        """
        Return the flat array of variational parameters
        that should be optimized.
        """
        return np.array([p.value for p in self.weights[0]])
