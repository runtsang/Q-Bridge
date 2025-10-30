import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, transpile, assemble

class SelfAttentionGen180:
    """
    Quantum implementation of the self‑attention module.  The API matches
    the classical version: run(rotation_params, entangle_params, inputs).
    Each input is a 1‑D array of length ``embed_dim`` (the number of qubits).
    """

    def __init__(
        self,
        embed_dim: int = 4,
        threshold: float = 0.0,
        shots: int = 1024,
    ):
        self.embed_dim = embed_dim
        self.threshold = threshold
        self.shots = shots
        self.backend = Aer.get_backend("qasm_simulator")

    def _build_circuit(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        input_data: np.ndarray,
    ) -> QuantumCircuit:
        """
        Construct a parameterised circuit that mirrors the classical
        self‑attention block but operates on a quantum register.
        """
        n = self.embed_dim
        qr = QuantumRegister(n, "q")
        cr = ClassicalRegister(n, "c")
        circuit = QuantumCircuit(qr, cr)

        # Encode the classical input as rotation angles (thresholded)
        for i in range(n):
            angle = np.pi if input_data[i] > self.threshold else 0.0
            circuit.rx(angle, i)

        # Rotation layer (three Euler angles per qubit)
        for i in range(n):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)

        # Entanglement layer (controlled‑RX)
        for i in range(n - 1):
            circuit.crx(entangle_params[i], i, i + 1)

        circuit.measure_all()
        return circuit

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
        shots: int | None = None,
    ) -> np.ndarray:
        """
        Execute the quantum self‑attention circuit for a batch of inputs.

        Parameters
        ----------
        rotation_params : np.ndarray
            Array of shape (3 * embed_dim,) containing Euler angles.
        entangle_params : np.ndarray
            Array of shape (embed_dim - 1,) for the controlled‑RX gates.
        inputs : np.ndarray
            Batch of 1‑D arrays of length ``embed_dim``.
        shots : int, optional
            Number of shots per circuit.  Defaults to the instance value.

        Returns
        -------
        np.ndarray
            Expectation value of Z for each qubit, averaged over shots.
        """
        if shots is None:
            shots = self.shots

        expectations = []
        for inp in inputs:
            circ = self._build_circuit(rotation_params, entangle_params, inp)
            compiled = transpile(circ, self.backend)
            qobj = assemble(compiled, shots=shots)
            job = self.backend.run(qobj)
            result = job.result().get_counts(circ)

            # Compute expectation of Z (|0> → +1, |1> → -1)
            exp = 0.0
            for bitstring, count in result.items():
                z = 1 - 2 * sum(int(b) for b in bitstring)
                exp += z * count
            exp /= shots
            expectations.append(exp)

        return np.array(expectations)

__all__ = ["SelfAttentionGen180"]
