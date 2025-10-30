"""Quantum implementation of the hybrid self‑attention block.

The circuit mirrors the classical flow:
- 3‑parameter rotations per qubit encode query/key/value angles.
- Controlled‑X gates create entanglement akin to self‑attention.
- A random layer injects non‑linear quantum features.
- Optional Ry gates driven by a fully‑connected parameter set.
- Measurement yields a classical expectation that serves as the output.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
from qiskit.circuit.library import RandomLayer


class HybridSelfAttention:
    """
    Quantum self‑attention block that emulates the classical
    HybridSelfAttention structure.
    """

    def __init__(self, n_qubits: int = 4) -> None:
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build_circuit(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        fc_params: np.ndarray | None = None,
    ) -> QuantumCircuit:
        """
        Constructs a parameterised circuit that maps the three groups of
        parameters to rotations, entanglement, and a random layer.
        """
        circuit = QuantumCircuit(self.qr, self.cr)

        # 3‑angle rotations per qubit (rx, ry, rz)
        for i in range(self.n_qubits):
            idx = 3 * i
            circuit.rx(rotation_params[idx], i)
            circuit.ry(rotation_params[idx + 1], i)
            circuit.rz(rotation_params[idx + 2], i)

        # Entanglement using CRX gates
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)

        # Random non‑linear layer (mimics a quantum kernel)
        random_layer = RandomLayer(num_qubits=self.n_qubits, ops_per_qubit=4)
        random_layer.apply(circuit)

        # Optional fully‑connected parameters as Ry rotations
        if fc_params is not None:
            for i, theta in enumerate(fc_params):
                if i < self.n_qubits:
                    circuit.ry(theta, i)

        # Measure all qubits
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(
        self,
        backend,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        fc_params: np.ndarray | None = None,
        shots: int = 1024,
    ):
        """
        Execute the circuit and return a scalar expectation value.

        Parameters
        ----------
        backend : qiskit.providers.base.Provider
            Execution backend.
        rotation_params : np.ndarray
            3 * n_qubits rotation angles.
        entangle_params : np.ndarray
            n_qubits - 1 controlled‑X angles.
        fc_params : np.ndarray, optional
            Parameters for the fully‑connected layer (Ry gates).
        shots : int, optional
            Number of shots for the measurement.

        Returns
        -------
        np.ndarray
            Array containing the expectation value of the Pauli‑Z
            measurement over all qubits.
        """
        circuit = self._build_circuit(rotation_params, entangle_params, fc_params)
        job = execute(circuit, backend, shots=shots)
        result = job.result()
        counts = result.get_counts(circuit)
        # Convert bitstring to integer and compute expectation
        exp_val = 0.0
        total = 0
        for bitstr, cnt in counts.items():
            val = int(bitstr, 2)
            exp_val += val * cnt
            total += cnt
        expectation = exp_val / total
        return np.array([expectation])


__all__ = ["HybridSelfAttention"]
