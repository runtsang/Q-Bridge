"""Hybrid quantum layer that mirrors a fully connected + self‑attention architecture.

The circuit applies a layer of rotations per qubit to emulate a fully connected
parameterized transform, followed by controlled‑rotations between adjacent qubits
to emulate a self‑attention style entanglement.  The expectation value of the
measured qubit string is returned as a proxy for the layer output.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

class HybridLayer:
    def __init__(self, n_qubits: int, backend=None, shots: int = 1024) -> None:
        self.n_qubits = n_qubits
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build_circuit(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> QuantumCircuit:
        """
        Build a quantum circuit that:
        * applies a 3‑parameter rotation (RX,RY,RZ) to each qubit,
        * entangles adjacent qubits with controlled‑RX gates,
        * measures all qubits.
        """
        circuit = QuantumCircuit(self.qr, self.cr)

        # Fully connected part: rotations per qubit
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)

        # Attention part: controlled rotations between neighbors
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)

        circuit.measure(self.qr, self.cr)
        return circuit

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> np.ndarray:
        """
        Execute the circuit and return the expectation value of the
        measured qubit string as a single‑dimensional output.
        """
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = qiskit.execute(circuit, self.backend, shots=self.shots)
        result = job.result().get_counts(circuit)

        # Convert measurement counts to expectation value
        probs = np.array(list(result.values()), dtype=float) / self.shots
        states = np.array([int(k, 2) for k in result.keys()], dtype=float)
        expectation = np.sum(states * probs)
        return np.array([expectation])

__all__ = ["HybridLayer"]
