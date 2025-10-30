"""Variational quantum circuit that consumes parameters produced by
the classical FraudDetectionHybrid model.  The ansatz consists of
Ry rotations on each qubit followed by a CX‑entangling layer and a
final measurement of Pauli‑Z on the first qubit.  The expectation
value returned by this circuit can be used as a fraud‑prediction
score."""
import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute


class FraudDetectionHybrid:
    """
    Quantum implementation that consumes a parameter vector produced by
    the classical FraudDetectionHybrid model.  The circuit is a variational
    ansatz consisting of Ry rotations followed by a controlled‑X entangling
    block and a final measurement of Pauli‑Z on the first qubit.  It returns
    the expectation value as a scalar.
    """

    def __init__(self, n_params: int, backend=None, shots: int = 1024):
        self.n_params = n_params
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.base_circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_params)
        # Entangling layer: CX between adjacent qubits
        for i in range(self.n_params - 1):
            qc.cx(i, i + 1)
        qc.barrier()
        return qc

    def run(self, params: np.ndarray) -> np.ndarray:
        if len(params)!= self.n_params:
            raise ValueError(f"Expected {self.n_params} parameters, got {len(params)}")
        qc = self.base_circuit.copy()
        for i, theta in enumerate(params):
            qc.ry(theta, i)
        qc.measure_all()
        job = execute(qc, self.backend, shots=self.shots)
        result = job.result()
        counts = result.get_counts(qc)
        expectation = 0.0
        for state, count in counts.items():
            # state string is in little‑endian order
            z = 1 if state[0] == "0" else -1
            expectation += z * count
        expectation /= self.shots
        return np.array([expectation])


__all__ = ["FraudDetectionHybrid"]
