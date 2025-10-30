import numpy as np
from qiskit import QuantumCircuit, Aer, execute
import torch

class SelfAttentionHybrid:
    """Quantum implementation of the hybrid self‑attention mechanism."""
    def __init__(self, n_qubits: int, backend=None, shots: int = 1024):
        self.n_qubits = n_qubits
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots

    def _build_circuit(self, q_angles: np.ndarray, k_angles: np.ndarray) -> QuantumCircuit:
        circuit = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            circuit.ry(q_angles[i], i)
            circuit.ry(k_angles[i], i)
        # Simple entanglement to correlate query and key
        circuit.cx(0, 1)
        circuit.barrier()
        circuit.measure_all()
        return circuit

    def run(self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            queries: torch.Tensor,
            keys: torch.Tensor) -> np.ndarray:
        scores = []
        for q, k in zip(queries, keys):
            q_angles = q.cpu().numpy()
            k_angles = k.cpu().numpy()
            circuit = self._build_circuit(q_angles, k_angles)
            job = execute(circuit, self.backend, shots=self.shots)
            result = job.result()
            counts = result.get_counts(circuit)
            exp = 0.0
            for outcome, cnt in counts.items():
                z = 1 if outcome[0] == '0' else -1
                exp += z * cnt
            exp /= self.shots
            scores.append(exp)
        return np.array(scores)
