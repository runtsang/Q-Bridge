import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit.library import RealAmplitudes
import numpy as np
from typing import Iterable

class QuantumFCL:
    """
    Parameterized quantum circuit that emulates a fully connected layer.
    It uses a RealAmplitudes ansatz with `n_qubits` parameters and returns
    the expectation value of the Pauli‑Z operator on each qubit.
    """
    def __init__(self, n_qubits: int = 1, shots: int = 1024):
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = Aer.get_backend("qasm_simulator")
        self.theta = qiskit.circuit.ParameterVector("theta", length=n_qubits)
        self._circuit = QuantumCircuit(n_qubits)
        self._circuit.append(RealAmplitudes(n_qubits, reps=1), range(n_qubits))
        self._circuit.measure_all()

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Evaluate the circuit for a list of parameter values.
        Returns a 1‑D array of Z‑expectation values, one per qubit.
        """
        if len(thetas)!= self.n_qubits:
            raise ValueError(f"Expected {self.n_qubits} parameters, got {len(thetas)}.")
        circ = self._circuit.copy()
        circ.assign_parameters({self.theta[i]: t for i, t in enumerate(thetas)}, inplace=True)
        job = execute(circ, self.backend, shots=self.shots)
        result = job.result()
        counts = result.get_counts(circ)
        probs = np.zeros((self.n_qubits, 2))
        for bitstring, count in counts.items():
            for i, bit in enumerate(reversed(bitstring)):
                probs[i, int(bit)] += count
        probs /= self.shots
        exp = probs[:, 0] - probs[:, 1]  # expectation of Z
        return exp

def QuantumFCL_factory(n_qubits: int = 1, shots: int = 1024) -> QuantumFCL:
    return QuantumFCL(n_qubits=n_qubits, shots=shots)

__all__ = ["QuantumFCL", "QuantumFCL_factory"]
