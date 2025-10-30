import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from typing import Iterable

class FCL:
    """
    Quantum fully connected layer using an entangled parameterized circuit.
    The ``run`` method accepts a list of parameters (thetas) and returns the
    expectation value of the sum of Z operators across all qubits.
    """
    def __init__(self, n_qubits: int = 4, backend=None, shots: int = 1024) -> None:
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = backend or Aer.get_backend('qasm_simulator')

        # Parameter vector
        self.theta = ParameterVector('theta', self.n_qubits)

        # Build circuit
        self.circuit = QuantumCircuit(self.n_qubits)
        # Initial Hadamard layer
        self.circuit.h(range(self.n_qubits))
        # Entangling layer: chain of CNOTs
        for i in range(self.n_qubits - 1):
            self.circuit.cx(i, i + 1)
        # Parameterized rotation layer
        for i, p in enumerate(self.theta):
            self.circuit.ry(p, i)
        # Final entangling layer (reverse)
        for i in reversed(range(self.n_qubits - 1)):
            self.circuit.cx(i, i + 1)
        # Measurement
        self.circuit.measure_all()

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Execute the circuit for the given parameters and return the expectation
        value of the sum of Z operators over all qubits.
        Parameters
        ----------
        thetas : Iterable[float]
            List of length n_qubits containing rotation angles.
        Returns
        -------
        np.ndarray
            Expectation value as a 1-element array.
        """
        if len(thetas)!= self.n_qubits:
            raise ValueError(f"Expected {self.n_qubits} parameters, got {len(thetas)}")
        bind = {self.theta[i]: thetas[i] for i in range(self.n_qubits)}
        circ = self.circuit.bind_parameters(bind)
        job = execute(circ, self.backend, shots=self.shots)
        result = job.result()
        counts = result.get_counts(circ)
        probs = {k: v / self.shots for k, v in counts.items()}

        # Compute expectation of sum of Z operators
        exp_val = 0.0
        for bitstring, prob in probs.items():
            # Qiskit orders bits as qubit n-1... 0
            bits = [int(b) for b in reversed(bitstring)]
            z_vals = [1.0 if b == 0 else -1.0 for b in bits]
            exp_val += prob * sum(z_vals)
        return np.array([exp_val])

__all__ = ["FCL"]
