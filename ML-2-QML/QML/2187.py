"""
Quantum self‑attention circuit with a variational entanglement topology
and a measurement‑based soft‑max output.

Parameters are passed in a dictionary:
    - `rot`   : (3 * n_qubits,)  – RX, RY, RZ angles per qubit
    - `ent`   : (n_qubits - 1,)  – angles for controlled‑RX entanglement
    - `topo`  : str, "linear" or "ring" specifying the entanglement pattern
"""
import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute

class QuantumSelfAttention:
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _prepare_state(self, circuit: QuantumCircuit, input_angles: np.ndarray):
        """Rotate each qubit by an angle derived from the input embedding."""
        for i, ang in enumerate(input_angles):
            circuit.ry(ang, i)

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray, topo: str):
        circuit = QuantumCircuit(self.qr, self.cr)

        # Parameterized single‑qubit rotations
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)

        # Entanglement pattern
        if topo == "ring":
            indices = list(range(self.n_qubits)) + [0]
        else:  # linear
            indices = list(range(self.n_qubits))

        for i in range(len(indices) - 1):
            circuit.crx(entangle_params[i], indices[i], indices[i + 1])

        return circuit

    def run(self, backend, rotation_params: np.ndarray, entangle_params: np.ndarray,
            input_embedding: np.ndarray, shots: int = 1024, topo: str = "linear"):
        """
        Execute the circuit and return a soft‑max probability vector over all bit‑strings.
        """
        # Build circuit with variational parameters
        circuit = self._build_circuit(rotation_params, entangle_params, topo)

        # Prepare state from input embedding
        self._prepare_state(circuit, input_embedding)

        # Measurement
        circuit.measure(self.qr, self.cr)

        # Execution
        job = execute(circuit, backend=backend, shots=shots)
        result = job.result()
        counts = result.get_counts(circuit)

        # Convert counts to probability vector
        num_states = 2 ** self.n_qubits
        probs = np.array([counts.get(f"{i:0{self.n_qubits}b}", 0) for i in range(num_states)]) / shots

        # Soft‑max over the probability vector
        sm = np.exp(probs) / np.exp(probs).sum()
        return sm
