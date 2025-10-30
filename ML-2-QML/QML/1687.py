"""Quantum classifier with configurable ansatz and measurement strategy.

The implementation builds a layered variational circuit that can be
used with Qiskit or Pennylane backends.  The design mirrors the
classical counterpart: the number of qubits is equal to the number of
features, the depth controls the number of variational layers, and
observable indices are provided for downstream loss computation.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.providers import BackendV1
from qiskit.providers.aer import QasmSimulator, StatevectorSimulator

# Optional: provide a fallback to pennylane if qiskit is unavailable
try:
    import pennylane as qml
except ImportError:
    qml = None  # type: ignore


class QuantumClassifierModel:
    """Variational circuit that mirrors the classical classifier.

    Parameters
    ----------
    num_qubits : int
        Number of qubits, equal to the number of input features.
    depth : int
        Number of variational layers.
    entanglement : str
        Entanglement pattern; one of ``'full'``, ``'circular'`` or ``'cnot'``.
    ansatz : str
        Single‑qubit gate used in each layer; supported values are
        ``'ry'`` and ``'rz'``.  The choice influences expressivity.
    """

    def __init__(
        self,
        num_qubits: int,
        depth: int,
        entanglement: str = "full",
        ansatz: str = "ry",
    ) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.entanglement = entanglement
        self.ansatz = ansatz
        self._build_circuit()

    # ------------------------------------------------------------------ #
    #  Circuit construction helpers
    # ------------------------------------------------------------------ #
    def _entangle(self, qc: QuantumCircuit) -> None:
        """Apply the chosen entanglement pattern."""
        if self.entanglement == "full":
            for i in range(self.num_qubits):
                for j in range(i + 1, self.num_qubits):
                    qc.cz(i, j)
        elif self.entanglement == "circular":
            for i in range(self.num_qubits):
                qc.cz(i, (i + 1) % self.num_qubits)
        elif self.entanglement == "cnot":
            for i in range(self.num_qubits - 1):
                qc.cx(i, i + 1)
        else:
            raise ValueError(f"Unsupported entanglement: {self.entanglement}")

    def _single_qubit_layer(self, qc: QuantumCircuit, param_vec: ParameterVector) -> None:
        """Apply a rotation layer on all qubits."""
        for q in range(self.num_qubits):
            gate = getattr(qc, self.ansatz)
            gate(param_vec[q], q)

    def _build_circuit(self) -> None:
        # Encoding parameters (data reuploading)
        self.encoding: List[ParameterVector] = [
            ParameterVector(f"x_{i}", 1) for i in range(self.num_qubits)
        ]

        # Variational parameters
        weight_params = ParameterVector(
            "theta", self.num_qubits * self.depth
        )

        self.circuit = QuantumCircuit(self.num_qubits)

        # Data reuploading: first layer of data encoding
        for i, pv in enumerate(self.encoding):
            self.circuit.rx(pv[0], i)

        # Iterated variational layers
        idx = 0
        for _ in range(self.depth):
            # Single‑qubit rotations
            for q in range(self.num_qubits):
                gate = getattr(self.circuit, self.ansatz)
                gate(weight_params[idx], q)
                idx += 1
            # Entanglement block
            self._entangle(self.circuit)

        # Observables: Z on each qubit
        self.observables: List[SparsePauliOp] = [
            SparsePauliOp(f"{'I' * i}Z{'I' * (self.num_qubits - i - 1)}")
            for i in range(self.num_qubits)
        ]

        # Parameter vector lists for external use
        self.params: List[ParameterVector] = [*self.encoding, weight_params]

    # ------------------------------------------------------------------ #
    #  Simulation helpers
    # ------------------------------------------------------------------ #
    def get_circuit(self) -> QuantumCircuit:
        """Return the underlying Qiskit circuit."""
        return self.circuit

    def get_observables(self) -> List[SparsePauliOp]:
        """Return the observable list for expectation evaluation."""
        return self.observables

    def simulate(
        self,
        backend: BackendV1 | str = "qasm_simulator",
        shots: int = 1024,
        seed: int | None = None,
    ) -> np.ndarray:
        """Execute the circuit on a Qiskit backend and return expectation values.

        Parameters
        ----------
        backend : BackendV1 | str
            Either a Qiskit backend instance or the name of a built‑in simulator.
        shots : int
            Number of shots for the qasm simulator.
        seed : int | None
            Random seed for reproducibility.
        """
        if isinstance(backend, str):
            if backend == "statevector_simulator":
                backend = StatevectorSimulator()
            else:
                backend = QasmSimulator()
        job = execute(self.circuit, backend, shots=shots, seed_simulator=seed)
        result = job.result()
        exp_vals = np.array(
            [
                result.get_expectation_value(obs, self.circuit)
                for obs in self.observables
            ]
        )
        return exp_vals

    # ------------------------------------------------------------------ #
    #  Pennylane interface (optional)
    # ------------------------------------------------------------------ #
    def to_pennylane(self) -> "qml.QNode":
        """Return a Pennylane QNode using the same circuit layout."""
        if qml is None:
            raise ImportError("Pennylane is not installed.")
        dev = qml.device("default.qubit", wires=self.num_qubits)

        def circuit(*weights):
            for i in range(self.num_qubits):
                qml.RX(weights[i], wires=i)
            for l in range(self.depth):
                for w in range(self.num_qubits):
                    gate = getattr(qml, self.ansatz.upper())
                    gate(weights[self.num_qubits * l + w], wires=w)
                # Entanglement
                if self.entanglement == "full":
                    for i in range(self.num_qubits):
                        for j in range(i + 1, self.num_qubits):
                            qml.CZ(wires=[i, j])
                elif self.entanglement == "circular":
                    for i in range(self.num_qubits):
                        qml.CZ(wires=[i, (i + 1) % self.num_qubits])
                elif self.entanglement == "cnot":
                    for i in range(self.num_qubits - 1):
                        qml.CNOT(wires=[i, i + 1])
                else:
                    raise ValueError(f"Unsupported entanglement: {self.entanglement}")

            return [qml.expval(qml.PauliZ(w)) for w in range(self.num_qubits)]

        return qml.QNode(circuit, dev)

__all__ = ["QuantumClassifierModel"]
