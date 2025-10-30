import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp
from typing import Iterable, Sequence, List

class SelfAttentionHybrid:
    """Quantum self‑attention circuit with a classification ansatz and fast estimation."""
    def __init__(self, n_qubits: int = 4, depth: int = 2, backend=None) -> None:
        self.n_qubits = n_qubits
        self.depth = depth
        self.backend = backend or qiskit.Aer.get_backend('qasm_simulator')
        self.observables = [SparsePauliOp('Z' + 'I' * (n_qubits - i - 1)) for i in range(n_qubits)]

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        # Encoding with RX, RY, RZ rotations
        for i in range(self.n_qubits):
            qc.rx(rotation_params[3 * i], i)
            qc.ry(rotation_params[3 * i + 1], i)
            qc.rz(rotation_params[3 * i + 2], i)
        # Self‑attention entanglement
        for i in range(self.n_qubits - 1):
            qc.crx(entangle_params[i], i, i + 1)
        # Classification ansatz
        idx = 0
        for _ in range(self.depth):
            for qubit in range(self.n_qubits):
                qc.ry(entangle_params[idx], qubit)
                idx += 1
            for qubit in range(self.n_qubits - 1):
                qc.cz(qubit, qubit + 1)
        return qc

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, shots: int = 1024):
        qc = self._build_circuit(rotation_params, entangle_params)
        job = qiskit.execute(qc, self.backend, shots=shots)
        return job.result().get_counts(qc)

    def evaluate(self,
                 observables: Iterable[SparsePauliOp],
                 parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        results: List[List[complex]] = []
        for params in parameter_sets:
            rot_len = self.n_qubits * 3
            rot = np.array(params[:rot_len])
            ent = np.array(params[rot_len:])
            qc = self._build_circuit(rot, ent)
            state = Statevector.from_instruction(qc)
            results.append([state.expectation_value(obs) for obs in observables])
        return results
