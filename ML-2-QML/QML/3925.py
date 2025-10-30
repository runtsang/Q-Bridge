import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer

class HybridQuantumAttentionLayerQuantum:
    """
    Quantum implementation of a self‑attention style block that mirrors the classical
    HybridQuantumAttentionLayer.  Inputs are encoded as Y‑rotations on each qubit,
    followed by parameterised Rx/Ry/Rz rotations (rotation_params) and controlled‑Rx
    entangling gates (entangle_params).  The expectation value of the Z operator
    on each qubit is returned as the output vector.
    """

    def __init__(self, n_qubits: int, backend=None, shots: int = 1024) -> None:
        self.n_qubits = n_qubits
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build_circuit(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)

        # Encode inputs as RY rotations
        for i in range(self.n_qubits):
            circuit.ry(inputs[i], i)

        # Apply rotation parameters (Rx,Ry,Rz) per qubit
        for i in range(self.n_qubits):
            idx = 3 * i
            circuit.rx(rotation_params[idx], i)
            circuit.ry(rotation_params[idx + 1], i)
            circuit.rz(rotation_params[idx + 2], i)

        # Entangle qubits with controlled‑Rx gates
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)

        circuit.measure(self.qr, self.cr)
        return circuit

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """
        Execute the quantum circuit and return the expectation value of Z on each qubit.

        Parameters
        ----------
        rotation_params : array-like
            Flattened array of length 3*n_qubits containing Rx,Ry,Rz angles per qubit.
        entangle_params : array-like
            Array of length n_qubits-1 containing controlled‑Rx angles between qubit pairs.
        inputs : array-like
            Array of length n_qubits with input angles for the initial RY encoding.

        Returns
        -------
        numpy.ndarray
            Expectation values of the Pauli‑Z operator on each qubit, shape (n_qubits,).
        """
        circuit = self._build_circuit(rotation_params, entangle_params, inputs)
        job = execute(circuit, self.backend, shots=self.shots)
        result = job.result()
        counts = result.get_counts(circuit)

        exp_vals = np.zeros(self.n_qubits)
        for state, cnt in counts.items():
            prob = cnt / self.shots
            bits = np.array([int(b) for b in reversed(state)])  # LSB first
            exp_vals += (1 - 2 * bits) * prob  # Z eigenvalues: |0>=+1, |1>=-1
        return exp_vals
