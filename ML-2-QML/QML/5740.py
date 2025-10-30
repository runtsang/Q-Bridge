import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute

class SelfAttention:
    """
    Variational quantum circuit implementing a self‑attention style block.
    """

    def __init__(self, n_qubits: int = 4, depth: int = 1):
        self.n_qubits = n_qubits
        self.depth = depth
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")
        self.backend = Aer.get_backend("qasm_simulator")

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        """
        Build a parameterised circuit.

        Parameters
        ----------
        rotation_params : np.ndarray
            Flat array of shape (n_qubits * 3 * depth,) containing RX,RY,RZ parameters.
        entangle_params : np.ndarray
            Flat array of shape ((n_qubits - 1) * depth,) containing CRX parameters.

        Returns
        -------
        QuantumCircuit
            The constructed circuit with measurements.
        """
        circuit = QuantumCircuit(self.qr, self.cr)

        rot_idx = 0
        ent_idx = 0
        for _ in range(self.depth):
            # Rotations on each qubit
            for q in range(self.n_qubits):
                circuit.rx(rotation_params[rot_idx], q); rot_idx += 1
                circuit.ry(rotation_params[rot_idx], q); rot_idx += 1
                circuit.rz(rotation_params[rot_idx], q); rot_idx += 1

            # Entangling CRX gates between neighbouring qubits
            for q in range(self.n_qubits - 1):
                circuit.crx(entangle_params[ent_idx], q, q + 1); ent_idx += 1

        circuit.measure(self.qr, self.cr)
        return circuit

    def run(self, backend, rotation_params: np.ndarray, entangle_params: np.ndarray, shots: int = 1024):
        """
        Execute the circuit on the provided backend.

        Parameters
        ----------
        backend : qiskit.providers.Backend
            Quantum backend to run the simulation.
        rotation_params : np.ndarray
            Rotation parameters as described in `_build_circuit`.
        entangle_params : np.ndarray
            Entanglement parameters as described in `_build_circuit`.
        shots : int, optional
            Number of measurement shots. Defaults to 1024.

        Returns
        -------
        dict
            Measurement outcome probabilities (counts).
        """
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = execute(circuit, backend=backend, shots=shots)
        return job.result().get_counts(circuit)

__all__ = ["SelfAttention"]
