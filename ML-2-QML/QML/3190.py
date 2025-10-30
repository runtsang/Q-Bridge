"""Quantum self‑attention built with Qiskit."""
import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister


class SelfAttention:
    """
    Quantum self‑attention building a parametric circuit.
    Parameters are NumPy arrays of rotation angles.
    """

    def __init__(self, n_qubits: int = 4, backend=None):
        """
        Args:
            n_qubits: number of qubits in the circuit.
            backend: Qiskit backend; defaults to Aer qasm simulator.
        """
        self.n_qubits = n_qubits
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build_circuit(
        self, rotation_params: np.ndarray, entangle_params: np.ndarray
    ) -> qiskit.QuantumCircuit:
        """
        Construct a circuit that applies parameterized rotations and
        an entanglement layer before measuring all qubits.
        """
        if rotation_params.size!= 3 * self.n_qubits:
            raise ValueError(
                f"rotation_params must contain 3 * n_qubits = {3 * self.n_qubits} angles"
            )
        if entangle_params.size!= self.n_qubits - 1:
            raise ValueError(
                f"entangle_params must contain {self.n_qubits - 1} angles"
            )

        circuit = QuantumCircuit(self.qr, self.cr)
        # Single‑qubit rotations
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)

        # Entanglement layer (CNOT ladder)
        for i in range(self.n_qubits - 1):
            circuit.cx(i, i + 1)
            circuit.cx(i + 1, i)

        # Measurement
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
        shots: int = 1024,
    ):
        """
        Execute the self‑attention circuit and return measurement counts.

        Args:
            rotation_params: array of rotation angles (length 3 * n_qubits).
            entangle_params: array of entanglement angles (unused in this simple design but kept for API compatibility).
            inputs: input data; shape is ignored but kept for API symmetry.
            shots: number of shots for the measurement.

        Returns:
            dict mapping bit‑strings to counts.
        """
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = qiskit.execute(circuit, self.backend, shots=shots)
        return job.result().get_counts(circuit)
