import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.providers.aer import AerSimulator

class SelfAttentionEnhanced:
    """
    Variational quantum self‑attention block.  The circuit is parameterised by
    rotation and entanglement angles.  A parameter‑shift rule is available
    for analytic gradients, making the module trainable in a hybrid loop.
    """

    def __init__(self, n_qubits: int = 4, backend=None):
        self.n_qubits = n_qubits
        self.backend = backend or AerSimulator()

    def _build_circuit(self,
                       rotation_vals: np.ndarray,
                       entangle_vals: np.ndarray) -> QuantumCircuit:
        """Return a compiled circuit with the supplied parameters."""
        qr = QuantumRegister(self.n_qubits, "q")
        cr = ClassicalRegister(self.n_qubits, "c")
        qc = QuantumCircuit(qr, cr)

        # Rotations
        for i in range(self.n_qubits):
            qc.rx(rotation_vals[3 * i], qr[i])
            qc.ry(rotation_vals[3 * i + 1], qr[i])
            qc.rz(rotation_vals[3 * i + 2], qr[i])

        # Entanglement
        for i in range(self.n_qubits - 1):
            qc.cx(qr[i], qr[i + 1])
            qc.rz(entangle_vals[i], qr[i + 1])

        qc.measure(qr, cr)
        return qc

    def run(self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            shots: int = 1024) -> np.ndarray:
        """
        Execute the circuit and return a measurement probability vector.
        :param rotation_params: array of shape (3*n_qubits,)
        :param entangle_params: array of shape (n_qubits-1,)
        :return: probability distribution over bitstrings as a 1‑D array.
        """
        qc = self._build_circuit(rotation_params, entangle_params)
        job = qiskit.execute(qc, self.backend, shots=shots)
        counts = job.result().get_counts(qc)
        probs = np.zeros(2 ** self.n_qubits)
        for bitstr, c in counts.items():
            idx = int(bitstr, 2)
            probs[idx] = c / shots
        return probs

    def expectation_z(self,
                      rotation_params: np.ndarray,
                      entangle_params: np.ndarray) -> np.ndarray:
        """
        Compute expectation values of Z on each qubit using the measurement
        distribution.  The expectation for qubit i is
        E[Z_i] = Σ (-1)^bit_i * P(bitstring).
        """
        probs = self.run(rotation_params, entangle_params)
        expz = []
        for i in range(self.n_qubits):
            exp = 0.0
            for idx, p in enumerate(probs):
                bit = (idx >> i) & 1
                exp += ((-1) ** bit) * p
            expz.append(exp)
        return np.array(expz)

    def parameter_shift_grad(self,
                             rotation_params: np.ndarray,
                             entangle_params: np.ndarray,
                             shift: float = np.pi / 2) -> dict:
        """
        Compute gradients of the expectation values w.r.t all parameters
        using the parameter‑shift rule.  Returns a dictionary mapping
        parameter names to gradient arrays of shape (param_size, n_qubits).
        """
        grads = {}
        # Gradient w.r.t rotation params
        grad_rot = np.zeros((rotation_params.size, self.n_qubits))
        for i in range(rotation_params.size):
            plus = rotation_params.copy()
            minus = rotation_params.copy()
            plus[i] += shift
            minus[i] -= shift
            f_plus = self.expectation_z(plus, entangle_params)
            f_minus = self.expectation_z(minus, entangle_params)
            grad_rot[i] = (f_plus - f_minus) / (2 * np.sin(shift))
        grads["rotation"] = grad_rot

        # Gradient w.r.t entangle params
        grad_ent = np.zeros((entangle_params.size, self.n_qubits))
        for i in range(entangle_params.size):
            plus = entangle_params.copy()
            minus = entangle_params.copy()
            plus[i] += shift
            minus[i] -= shift
            f_plus = self.expectation_z(rotation_params, plus)
            f_minus = self.expectation_z(rotation_params, minus)
            grad_ent[i] = (f_plus - f_minus) / (2 * np.sin(shift))
        grads["entangle"] = grad_ent
        return grads

__all__ = ["SelfAttentionEnhanced"]
