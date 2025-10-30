"""Variational quantum classifier with configurable ansatz and backend support."""

from __future__ import annotations

from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.algorithms import Sampler


def build_classifier_circuit(
    num_qubits: int,
    depth: int,
) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
    """Construct a layered variational circuit.

    Returns:
        circuit: QuantumCircuit with symbolic parameters
        encoding: list of ParameterVector for data encoding
        weights: list of ParameterVector for variational parameters
        observables: list of SparsePauliOp to be measured
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    qc = QuantumCircuit(num_qubits)
    for idx, qubit in enumerate(range(num_qubits)):
        qc.rx(encoding[idx], qubit)

    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            qc.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            qc.cz(qubit, qubit + 1)

    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]
    return qc, [encoding], [weights], observables


class QuantumClassifierModel:
    """Variational quantum classifier exposing a predict method."""
    def __init__(
        self,
        num_qubits: int,
        depth: int,
        backend: str = "aer_simulator_statevector",
    ):
        self.num_qubits = num_qubits
        self.depth = depth
        self.backend = backend
        self.circuit, self.encoding, self.weights, self.observables = build_classifier_circuit(
            num_qubits, depth
        )
        self.sampler = Sampler(backend=self.backend)

    def _bind_circuit(self, data: List[float], params: List[float]) -> QuantumCircuit:
        """Bind classical data and variational parameters to the symbolic circuit."""
        bind_dict = {p.name(): v for p, v in zip(self.encoding[0], data)}
        bind_dict.update({p.name(): v for p, v in zip(self.weights[0], params)})
        return self.circuit.bind_parameters(bind_dict)

    def predict(self, data: List[float], params: List[float]) -> List[float]:
        """Return expectation values of Z on each qubit as logits."""
        bound_circuit = self._bind_circuit(data, params)
        result = self.sampler.run(bound_circuit, observables=self.observables).result()
        exp_vals = result.get_expectation_values()
        return exp_vals.tolist()

    def get_weight_sizes(self) -> List[int]:
        """Return number of trainable parameters."""
        return [len(w) for w in self.weights]


__all__ = ["build_classifier_circuit", "QuantumClassifierModel"]
