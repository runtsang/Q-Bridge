"""Advanced variational classifier with data re‑uploading and dual entangling layers."""

from __future__ import annotations

from typing import List, Tuple

from qiskit import Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import QuantumCircuit


class QuantumClassifierModel:
    """
    A parameter‑ized quantum circuit for binary classification.
    Supports flexible backend selection, shot counts and dual entangling layers.
    """

    __all__ = ["QuantumClassifierModel"]

    def __init__(
        self,
        num_qubits: int,
        depth: int = 3,
        backend: str = "qasm_simulator",
        shots: int = 1024,
        backend_options: dict | None = None,
    ) -> None:
        """
        Parameters
        ----------
        num_qubits : int
            Number of qubits (features).
        depth : int, default=3
            Number of variational layers.
        backend : str, default="qasm_simulator"
            Qiskit backend name (e.g., 'qasm_simulator','statevector_simulator').
        shots : int, default=1024
            Number of sampling shots for expectation values.
        backend_options : dict | None, default=None
            Optional backend configuration options.
        """
        self.num_qubits = num_qubits
        self.depth = depth
        self.shots = shots
        self.backend_options = backend_options or {}

        # Parameter vectors
        self.encoding = ParameterVector("x", num_qubits)
        self.weights = ParameterVector("theta", num_qubits * depth)

        # Build circuit
        self.circuit, self.param_names, self.weight_names, self.observables = self._build_circuit()

        # Backend
        self.backend = Aer.get_backend(backend)

    def _build_circuit(self) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
        """Construct a layered ansatz with data re‑uploading and dual entangling."""
        qc = QuantumCircuit(self.num_qubits)

        # Data encoding
        for qubit, param in enumerate(self.encoding):
            qc.rx(param, qubit)

        idx = 0
        for _ in range(self.depth):
            # Rotation layer
            for qubit in range(self.num_qubits):
                qc.ry(self.weights[idx], qubit)
                idx += 1

            # Primary entangling layer (CZ)
            for qubit in range(self.num_qubits - 1):
                qc.cz(qubit, qubit + 1)

            # Secondary entangling layer (CNOT) for richer connectivity
            for qubit in range(self.num_qubits - 1):
                qc.cx(qubit, qubit + 1)

        # Observables: Z on each qubit
        observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (self.num_qubits - i - 1))
            for i in range(self.num_qubits)
        ]

        return qc, list(self.encoding), list(self.weights), observables

    def execute(self, data: List[List[float]]) -> List[List[float]]:
        """
        Execute the circuit on a batch of data points.

        Parameters
        ----------
        data : List[List[float]]
            Batch of feature vectors (length = num_qubits).

        Returns
        -------
        List[List[float]]
            Expectation values for each observable per data point.
        """
        results: List[List[float]] = []

        for datum in data:
            # Bind parameters
            param_dict = {str(p): v for p, v in zip(self.encoding, datum)}
            bound_circuit = self.circuit.bind_parameters(param_dict)

            # Run
            job = execute(bound_circuit, self.backend, shots=self.shots, **self.backend_options)
            result = job.result()

            # Extract expectation values
            exp_vals = [result.get_expectation_value(op, bound_circuit) for op in self.observables]
            results.append(exp_vals)

        return results

    # Auxiliary introspection methods

    def get_encoding(self) -> List[ParameterVector]:
        """Parameter vector used for data encoding."""
        return self.encoding

    def get_weight_sizes(self) -> List[int]:
        """Total number of variational parameters."""
        return [self.weights.size]

    def get_observables(self) -> List[SparsePauliOp]:
        """Observables returned by the circuit."""
        return self.observables
