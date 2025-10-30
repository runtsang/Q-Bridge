from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from typing import Iterable, Tuple, List

class QuantumClassifier:
    """Quantum circuit builder that mirrors the classical interface.

    The circuit consists of a data‑encoding layer followed by `depth` variational
    layers.  An arbitrary entanglement pattern can be chosen (linear, full, or
    custom).  The returned tuple contains the circuit, the encoding parameters,
    the variational parameters, and the Pauli observables for expectation
    value estimation.
    """
    def __init__(self, num_qubits: int, depth: int = 3,
                 entanglement: str = "linear", encoding: str = "rx"):
        self.num_qubits = num_qubits
        self.depth = depth
        self.entanglement = entanglement
        self.encoding = encoding

    def build(self) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
        enc_params = ParameterVector("x", self.num_qubits)
        var_params = ParameterVector("theta", self.num_qubits * self.depth)

        qc = QuantumCircuit(self.num_qubits)

        # Data‑encoding
        if self.encoding == "rx":
            for q, p in zip(range(self.num_qubits), enc_params):
                qc.rx(p, q)
        elif self.encoding == "ry":
            for q, p in zip(range(self.num_qubits), enc_params):
                qc.ry(p, q)
        else:  # default to rx
            for q, p in zip(range(self.num_qubits), enc_params):
                qc.rx(p, q)

        # Variational layers
        idx = 0
        for _ in range(self.depth):
            for q in range(self.num_qubits):
                qc.ry(var_params[idx], q)
                idx += 1
            # Entanglement pattern
            if self.entanglement == "full":
                for i in range(self.num_qubits):
                    for j in range(i + 1, self.num_qubits):
                        qc.cz(i, j)
            elif self.entanglement == "linear":
                for i in range(self.num_qubits - 1):
                    qc.cz(i, i + 1)
            else:  # custom patterns can be added
                for i in range(self.num_qubits - 1):
                    qc.cz(i, i + 1)

        # Observables
        observables = [SparsePauliOp("I" * i + "Z" + "I" * (self.num_qubits - i - 1))
                       for i in range(self.num_qubits)]
        return qc, list(enc_params), list(var_params), observables

    @staticmethod
    def build_classifier_circuit(num_qubits: int, depth: int = 3,
                                 entanglement: str = "linear",
                                 encoding: str = "rx") -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
        """Convenience wrapper that matches the classical API."""
        return QuantumClassifier(num_qubits, depth, entanglement, encoding).build()

__all__ = ["QuantumClassifier"]
