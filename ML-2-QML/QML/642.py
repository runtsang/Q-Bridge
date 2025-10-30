"""Qiskit‑based quantum self‑attention circuit.

The class exposes a minimal API that builds a parameterized circuit
with `n_qubits` qubits.  It returns a probability distribution over
the qubits by measuring the expectation value of Pauli‑Z on each
qubit and mapping the result to [0, 1] via a sigmoid‑like transform.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.opflow import PauliExpectation, StateFn, CircuitStateFn, PauliOp, Pauli

class QuantumSelfAttention:
    def __init__(self, n_qubits: int, backend=None):
        self.n_qubits = n_qubits
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        qr = QuantumRegister(self.n_qubits, "q")
        circuit = QuantumCircuit(qr)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        return circuit

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> np.ndarray:
        """Return a probability distribution over the `n_qubits` qubits."""
        circuit = self._build_circuit(rotation_params, entangle_params)

        probs = []
        for i in range(self.n_qubits):
            pauli_str = ["I"] * self.n_qubits
            pauli_str[i] = "Z"
            op = PauliOp(Pauli("".join(pauli_str)))
            state = StateFn(op, wrap=True)
            circuit_state = CircuitStateFn(circuit)
            exp_val = PauliExpectation().convert(state @ circuit_state)
            result = exp_val.eval(backend=self.backend)
            prob = 0.5 * (result + 1.0)
            probs.append(prob)

        probs = np.array(probs)
        probs = probs / probs.sum()  # normalize to sum to 1
        return probs

__all__ = ["QuantumSelfAttention"]
