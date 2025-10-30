import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import Statevector

class EstimatorQNNGen128:
    def __init__(self, num_qubits=8):
        self.num_qubits = num_qubits
        self.backend = Aer.get_backend('statevector_simulator')

    def _amplitude_encode(self, vector: np.ndarray) -> QuantumCircuit:
        vec = vector / np.linalg.norm(vector)
        qc = QuantumCircuit(self.num_qubits)
        qc.initialize(vec, qc.qubits)
        return qc

    def _build_ansatz(self) -> QuantumCircuit:
        ansatz = RealAmplitudes(self.num_qubits, reps=3)
        qc = QuantumCircuit(self.num_qubits)
        qc.compose(ansatz, inplace=True)
        return qc

    def evaluate(self, x: np.ndarray, shots=1024) -> float:
        if x.size!= 2 ** self.num_qubits:
            raise ValueError(f"Input vector must have size {2 ** self.num_qubits}")
        qc = self._amplitude_encode(x)
        qc.compose(self._build_ansatz(), inplace=True)
        qc.measure_all()
        job = execute(qc, self.backend, shots=shots)
        result = job.result()
        counts = result.get_counts(qc)
        exp = 0.0
        for bitstring, count in counts.items():
            z = 1 if bitstring[-1] == '0' else -1
            exp += z * count
        exp /= shots
        return exp

def EstimatorQNN():
    return EstimatorQNNGen128(num_qubits=8)

__all__ = ["EstimatorQNNGen128", "EstimatorQNN"]
