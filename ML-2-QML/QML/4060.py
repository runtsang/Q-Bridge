import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from typing import Iterable, List, Sequence, Dict, Any

class HybridAttentionLayer:
    """Quantum hybrid layer: a fully connected rotation block followed by a self‑attention style entanglement."""
    def __init__(self, n_qubits: int, backend=None, shots: int = 1024):
        self.n_qubits = n_qubits
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots

    def _build_circuit(
        self,
        fc_params: np.ndarray,
        attn_rot: np.ndarray,
        attn_ent: np.ndarray,
    ) -> QuantumCircuit:
        qr = QuantumRegister(self.n_qubits, "q")
        cr = ClassicalRegister(self.n_qubits, "c")
        circuit = QuantumCircuit(qr, cr)

        # Fully‑connected rotation block (Ry gates)
        for i, theta in enumerate(fc_params):
            circuit.ry(theta, qr[i])

        # Self‑attention entanglement block
        for i, rot in enumerate(attn_rot):
            circuit.rx(rot[0], qr[i])
            circuit.ry(rot[1], qr[i])
            circuit.rz(rot[2], qr[i])

        for i in range(self.n_qubits - 1):
            circuit.crx(attn_ent[i], qr[i], qr[i + 1])

        circuit.measure(qr, cr)
        return circuit

    def run(
        self,
        params: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Execute the circuit with the supplied parameters and return the expectation of Pauli‑Z on qubit 0."""
        circuit = self._build_circuit(
            params["fc_params"],
            params["attention_rot"],
            params["attention_entangle"],
        )
        job = qiskit.execute(circuit, self.backend, shots=self.shots)
        result = job.result()
        counts = result.get_counts(circuit)
        probs = np.array(list(counts.values())) / self.shots
        states = np.array([int(k, 2) for k in counts.keys()])
        # Convert binary string to bitstring value of qubit 0
        qubit0 = states & 1
        expectation = np.sum((2 * qubit0 - 1) * probs)
        return np.array([expectation])

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Dict[str, np.ndarray]],
    ) -> List[List[complex]]:
        """Compute expectation values for each observable across all parameter sets."""
        results: List[List[complex]] = []
        for params in parameter_sets:
            state = Statevector.from_instruction(
                self._build_circuit(
                    params["fc_params"],
                    params["attention_rot"],
                    params["attention_entangle"],
                )
            )
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

__all__ = ["HybridAttentionLayer"]
