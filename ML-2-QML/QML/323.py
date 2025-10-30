import numpy as np
import qiskit
from qiskit.circuit import ParameterVector
from qiskit.providers.aer import AerSimulator
from typing import Iterable

class FCLayer:
    """
    Quantum implementation of a fully connected layer using a hardware‑efficient ansatz.
    The circuit consists of a configurable number of entangling layers; each layer
    applies Ry rotations followed by a CNOT chain.  Parameters are supplied as a flat
    list of angles and are bound to the circuit before execution.  The returned
    expectation is the expectation value of Pauli‑Z on the last qubit.
    """
    def __init__(self, n_qubits: int = 1, layers: int = 1, shots: int = 1000):
        self.n_qubits = n_qubits
        self.layers = layers
        self.shots = shots
        self.backend = AerSimulator(method="statevector")
        self._build_circuit()

    def _build_circuit(self):
        self.circuit = qiskit.QuantumCircuit(self.n_qubits)
        # total number of parameters
        self.n_params = self.n_qubits * self.layers
        self.params = ParameterVector("theta", self.n_params)
        idx = 0
        for _ in range(self.layers):
            # Ry rotations
            for q in range(self.n_qubits):
                self.circuit.ry(self.params[idx], q)
                idx += 1
            # CNOT chain
            for q in range(self.n_qubits - 1):
                self.circuit.cx(q, q + 1)
        # measurement of all qubits for expectation
        self.circuit.measure_all()

    def _expectation_from_counts(self, counts):
        """Compute expectation value of Z on the last qubit from measurement counts."""
        exp = 0.0
        total = 0
        for bitstring, cnt in counts.items():
            # bitstring is a string like '01'
            last_bit = int(bitstring[-1])
            # Z eigenvalue: +1 for |0>, -1 for |1>
            exp += (1 if last_bit == 0 else -1) * cnt
            total += cnt
        return exp / total

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Execute the circuit with the supplied rotation angles.
        Parameters
        ----------
        thetas : Iterable[float]
            Flat list of rotation angles matching the number of parameters.
        Returns
        -------
        np.ndarray
            Single‑element array containing the expectation value of Pauli‑Z.
        """
        if len(thetas)!= self.n_params:
            raise ValueError(f"Expected {self.n_params} parameters, got {len(thetas)}")
        bound_circuit = self.circuit.bind_parameters(
            {self.params[i]: theta for i, theta in enumerate(thetas)}
        )
        job = self.backend.run(bound_circuit, shots=self.shots)
        result = job.result()
        counts = result.get_counts(bound_circuit)
        expectation = self._expectation_from_counts(counts)
        return np.array([expectation], dtype=np.float32)

__all__ = ["FCLayer"]
