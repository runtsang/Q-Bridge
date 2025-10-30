"""Quantum hybrid sampler/classifier implementation.

The quantum side mirrors the classical architecture.  It builds a
parameterized circuit that can be used either as a sampler or a
classifier.  The interface exposes the same metadata as the
classical module so that parameters can be transferred.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

class HybridSamplerClassifier:
    """
    Quantum implementation of the hybrid sampler/classifier.

    Parameters
    ----------
    num_qubits: int
        Number of qubits (equal to the input feature dimension).
    depth: int
        Number of variational layers for the classifier mode.
    mode: str
        Either ``"sampler"`` or ``"classifier"``.
    """

    def __init__(self, num_qubits: int, depth: int = 1, mode: str = "sampler") -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.mode = mode.lower()

        if self.mode not in {"sampler", "classifier"}:
            raise ValueError(f"Unsupported mode {mode!r}")

        # Build the circuit and metadata
        self.circuit, self.encoding, self.weights, self.observables = self._build_circuit()

        # Metadata
        self.encoding_indices = list(range(num_qubits))
        self.weight_sizes = self._compute_weight_sizes()

    def _build_circuit(self) -> Tuple[QuantumCircuit, Iterable[ParameterVector], Iterable[ParameterVector], List[SparsePauliOp]]:
        """Construct a parameterized circuit with encoding and variational layers."""
        encoding = ParameterVector("x", self.num_qubits)
        weights = ParameterVector("theta", self.num_qubits * self.depth)

        qc = QuantumCircuit(self.num_qubits)

        # Data encoding
        for param, qubit in zip(encoding, range(self.num_qubits)):
            qc.rx(param, qubit)

        # Variational layers
        idx = 0
        for _ in range(self.depth):
            for qubit in range(self.num_qubits):
                qc.ry(weights[idx], qubit)
                idx += 1
            # Entangling layer
            for qubit in range(self.num_qubits - 1):
                qc.cz(qubit, qubit + 1)

        # Observables
        if self.mode == "classifier":
            observables = [SparsePauliOp("I" * i + "Z" + "I" * (self.num_qubits - i - 1))
                           for i in range(self.num_qubits)]
        else:  # sampler
            observables = [SparsePauliOp("Z" * self.num_qubits)]  # single global Z

        return qc, list(encoding), list(weights), observables

    def _compute_weight_sizes(self) -> List[int]:
        """Return the number of parameters per variational layer."""
        sizes = []
        # encoding parameters
        sizes.append(self.num_qubits)  # one per qubit
        # variational parameters per depth
        for _ in range(self.depth):
            sizes.append(self.num_qubits)  # one RY per qubit
        return sizes

    # Compatibility helpers
    def get_circuit(self) -> QuantumCircuit:
        """Return the underlying quantum circuit."""
        return self.circuit

    def get_encoding(self) -> List[ParameterVector]:
        """Return the encoding parameters."""
        return self.encoding

    def get_weights(self) -> List[ParameterVector]:
        """Return the variational parameters."""
        return self.weights

    def get_observables(self) -> List[SparsePauliOp]:
        """Return the measurement observables."""
        return self.observables

    def get_weight_sizes(self) -> List[int]:
        """Return the number of parameters per layer."""
        return self.weight_sizes

__all__ = ["HybridSamplerClassifier"]
