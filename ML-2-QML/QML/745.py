import numpy as np
import qiskit
from typing import Iterable

class FullyConnectedLayer:
    """
    Parameterised quantum circuit acting as a fully connected layer.
    Supports per‑qubit parameters and optional entanglement.
    """
    def __init__(self, n_qubits: int = 1, backend=None, shots: int = 100, entangle: bool = True):
        self.n_qubits = n_qubits
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.entangle = entangle

        self._circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta_{i}") for i in range(n_qubits)]

        # Circuit definition
        self._circuit.h(range(n_qubits))
        if entangle:
            for i in range(n_qubits - 1):
                self._circuit.cx(i, i + 1)
        for i, th in enumerate(self.theta):
            self._circuit.ry(th, i)
        self._circuit.measure_all()

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Execute the circuit with a list of parameters, one per qubit.
        Returns the expectation value of the Pauli‑Z observable summed over qubits.
        """
        if len(thetas)!= self.n_qubits:
            raise ValueError(f"Expected {self.n_qubits} parameters, got {len(thetas)}.")
        param_bind = {self.theta[i]: thetas[i] for i in range(self.n_qubits)}
        job = qiskit.execute(
            self._circuit,
            backend=self.backend,
            shots=self.shots,
            parameter_binds=[param_bind]
        )
        result = job.result()
        counts = result.get_counts(self._circuit)
        probs = np.array([counts.get(bit, 0) for bit in sorted(counts.keys())]) / self.shots
        states = np.array([int(bit, 2) for bit in sorted(counts.keys())])
        expectation = np.sum(states * probs)
        return np.array([expectation])

def FCL():
    """
    Factory that returns an instance of the quantum fully connected layer.
    """
    return FullyConnectedLayer()

__all__ = ["FullyConnectedLayer"]
