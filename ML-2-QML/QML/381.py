"""QuantumClassifier: a variational circuit that mirrors the classical interface.

The class builds a data‑re‑uploading ansatz with an arbitrary number of
re‑upload layers.  Encoding parameters are exposed as a ParameterVector,
while trainable variational angles are gathered in a separate vector.  The
public API matches that of the classical counterpart: ``build`` returns
the circuit, encoding indices, weight vector, and a list of Z‑observables."""
from __future__ import annotations

from typing import Iterable, List, Tuple
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.providers.aer import AerSimulator


class QuantumClassifier:
    """Variational circuit for binary classification with data‑re‑uploading."""

    def __init__(
        self,
        num_qubits: int,
        depth: int,
        reupload: int = 1,
        backend_name: str | None = None,
    ) -> None:
        """
        Parameters
        ----------
        num_qubits:
            Number of qubits (input dimension).
        depth:
            Number of variational layers per re‑upload block.
        reupload:
            How many times to re‑upload the data into the circuit.
        backend_name:
            Name of the Aer simulator to use.  If ``None`` defaults to
            ``AerSimulator()``.
        """
        self.num_qubits = num_qubits
        self.depth = depth
        self.reupload = reupload
        self.backend = AerSimulator() if backend_name is None else AerSimulator(backend_name)
        self.circuit, self.encoding, self.weights, self.observables = self.build()

    def build(
        self,
    ) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
        """Return the circuit, encoding indices, trainable weights and measurement operators."""
        encoding = ParameterVector("x", self.num_qubits)
        weights = ParameterVector("theta", self.num_qubits * self.depth * self.reupload)

        circuit = QuantumCircuit(self.num_qubits)
        # Initial data encoding
        for qubit, param in enumerate(encoding):
            circuit.rx(param, qubit)

        # Re‑uploading layers
        weight_idx = 0
        for _ in range(self.reupload):
            for _ in range(self.depth):
                # Rotations
                for qubit in range(self.num_qubits):
                    circuit.ry(weights[weight_idx], qubit)
                    weight_idx += 1
                # Entangling
                for qubit in range(self.num_qubits - 1):
                    circuit.cz(qubit, qubit + 1)
            # Re‑encode data before next block
            for qubit, param in enumerate(encoding):
                circuit.rx(param, qubit)

        # Measurement operators
        observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (self.num_qubits - i - 1))
            for i in range(self.num_qubits)
        ]

        return circuit, [encoding], [weights], observables

    def expectation_values(self, data: List[float]) -> List[float]:
        """Return expectation values of the observables for a given input vector."""
        bound_circuit = self.circuit.bind_parameters(
            {p: val for p, val in zip(self.encoding[0], data)}
        )
        result = self.backend.run(bound_circuit, shots=1, method="statevector").result()
        statevec = result.get_statevector()
        expectations = []
        for i in range(self.num_qubits):
            exp = 0.0
            for idx, amp in enumerate(statevec):
                bit = (idx >> i) & 1
                exp += ((-1) ** bit) * abs(amp) ** 2
            expectations.append(float(exp))
        return expectations

    def __repr__(self) -> str:
        return f"<QuantumClassifier qubits={self.num_qubits} depth={self.depth} reupload={self.reupload}>"
