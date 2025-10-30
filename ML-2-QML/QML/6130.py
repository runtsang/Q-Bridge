import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter

class HybridQuantumLayer:
    """
    Quantum implementation of the hybrid layer.  Each 2×2 image patch
    is encoded into a 4‑qubit circuit via Ry rotations, followed by a
    fixed entanglement layer.  The expectation value of the Z operator
    is measured for each qubit, aggregated across all patches, and a
    tanh non‑linearity is applied to produce the final scalar output.
    """
    def __init__(self, n_qubits: int = 4, shots: int = 1024) -> None:
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = Aer.get_backend('aer_simulator')

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Parameters
        ----------
        thetas : Iterable[float]
            28×28 image flattened into a 784‑element sequence.
        Returns
        -------
        np.ndarray
            Scalar output after quantum encoding and tanh.
        """
        thetas = np.array(thetas, dtype=np.float64)
        if thetas.size!= 28*28:
            raise ValueError(f"Expected 784 elements, got {thetas.size}")
        # Reshape into 14×14 patches of 4 values each
        patches = thetas.reshape(14, 14, 4).reshape(-1, 4)
        expectations = []
        for patch in patches:
            qc = QuantumCircuit(self.n_qubits)
            for i, val in enumerate(patch):
                qc.ry(val, i)
            # Simple entanglement: CNOT chain
            for i in range(self.n_qubits - 1):
                qc.cx(i, i + 1)
            qc.measure_all()
            job = execute(qc, self.backend, shots=self.shots)
            result = job.result()
            counts = result.get_counts(qc)
            probs = np.array(list(counts.values())) / self.shots
            states = np.array([int(k, 2) for k in counts.keys()])
            expectation = np.sum(states * probs)
            expectations.append(expectation)
        total = np.mean(expectations)
        return np.array([np.tanh(total)])
