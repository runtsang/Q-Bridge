import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute

class QuantumCircuitWrapper:
    """
    Simple 1‑qubit parameterised circuit returning the Z‑expectation.
    The first weight is treated as a rotation angle; the remaining
    weights are ignored but kept for API compatibility with the sampler.
    """
    def __init__(self, shots: int = 1024) -> None:
        self.shots = shots
        self.backend = Aer.get_backend("qasm_simulator")

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        thetas : np.ndarray
            Array of length 4; only the first element is used as the
            rotation angle for an Ry gate.

        Returns
        -------
        np.ndarray
            1‑D array containing the expectation value of Z.
        """
        theta = float(thetas[0])  # ensure scalar
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.ry(theta, 0)
        qc.measure_all()
        job = execute(qc, self.backend, shots=self.shots, memory=True)
        result = job.result()
        counts = result.get_counts(qc)
        # Convert measurement outcomes to 0/1 bits
        states = np.array([int(k, 2) for k in counts.keys()], dtype=int)
        probs = np.array(list(counts.values()), dtype=float) / self.shots
        expectation = np.sum((1 - 2 * states) * probs)  # Z = |0⟩⟨0| - |1⟩⟨1|
        return np.array([expectation])

__all__ = ["QuantumCircuitWrapper"]
