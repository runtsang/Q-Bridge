"""Quantum classifier circuit factory with richer entanglement patterns.

Features:
- Multiple entanglement topologies (cnot, cx, cz, chain, full).
- Configurable encoding gate (rx, ry, rz).
- Parameterized rotation layers with optional shift‑in‑phase entanglement.
- Returns circuit, encoding vector, weight vector, and observable list.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


class QuantumClassifierModel:
    """
    Quantum circuit that mirrors the classical factory interface.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    depth : int
        Number of variational layers.
    entanglement : str, default "cnot"
        Entanglement strategy. Options: 'cnot', 'cx', 'cz', 'chain', 'full'.
    encode_gate : str, default "ry"
        Single‑qubit gate used for feature encoding.
    ent_gate : str, default "cx"
        Two‑qubit gate used for entanglement.
    """

    def __init__(
        self,
        num_qubits: int,
        depth: int,
        entanglement: str = "cnot",
        encode_gate: str = "ry",
        ent_gate: str = "cx",
    ) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.entanglement = entanglement
        self.encode_gate = encode_gate
        self.ent_gate = ent_gate

    def _apply_entanglement(self, circuit: QuantumCircuit) -> None:
        """Apply entanglement according to the chosen strategy."""
        if self.entanglement == "cnot":
            for q in range(self.num_qubits):
                circuit.cx(q, (q + 1) % self.num_qubits)
        elif self.entanglement == "cx":
            for q in range(self.num_qubits - 1):
                circuit.cx(q, q + 1)
        elif self.entanglement == "cz":
            for q in range(self.num_qubits - 1):
                circuit.cz(q, q + 1)
        elif self.entanglement == "chain":
            for q in range(self.num_qubits - 1):
                circuit.cx(q, q + 1)
        elif self.entanglement == "full":
            for q1 in range(self.num_qubits):
                for q2 in range(q1 + 1, self.num_qubits):
                    circuit.cx(q1, q2)
        else:
            raise ValueError(f"Unknown entanglement strategy: {self.entanglement}")

    def build(self) -> Tuple[QuantumCircuit, Iterable[ParameterVector], Iterable[ParameterVector], List[SparsePauliOp]]:
        """
        Construct a layered ansatz with explicit feature encoding and variational parameters.

        Returns
        -------
        circuit : qiskit.QuantumCircuit
            The constructed circuit.
        encoding : list[ParameterVector]
            Parameters used for data encoding.
        weights : list[ParameterVector]
            Variational parameters of the circuit.
        observables : list[qiskit.quantum_info.SparsePauliOp]
            Pauli‑Z measurements on each qubit.
        """
        encoding = ParameterVector("x", self.num_qubits)
        weights = ParameterVector("theta", self.num_qubits * self.depth)

        circuit = QuantumCircuit(self.num_qubits)

        # Feature encoding
        for qubit, param in enumerate(encoding):
            getattr(circuit, self.encode_gate)(param, qubit)

        # Variational layers
        weight_idx = 0
        for _ in range(self.depth):
            # Single‑qubit rotations
            for qubit in range(self.num_qubits):
                getattr(circuit, self.encode_gate)(
                    weights[weight_idx], qubit
                )
                weight_idx += 1

            # Entanglement
            self._apply_entanglement(circuit)

        # Observables
        observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (self.num_qubits - i - 1))
            for i in range(self.num_qubits)
        ]

        return circuit, [encoding], [weights], observables
