import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer

class SelfAttention__gen260:
    """
    Quantum self‑attention block that:
    1. Splits each 2×2 image patch into a 4‑qubit register.
    2. Applies rotation gates (rotation_params) and entangling CRX gates (entangle_params).
    3. Measures all qubits and uses the measurement histogram to compute a weight for each patch.
    4. Produces a weighted sum of the original patch values, emulating attention.
    """
    def __init__(self, n_qubits: int = 4, patch_size: int = 2):
        self.n_qubits = n_qubits
        self.patch_size = patch_size
        self.backend = Aer.get_backend("qasm_simulator")

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        qr = QuantumRegister(self.n_qubits, "q")
        cr = ClassicalRegister(self.n_qubits, "c")
        qc = QuantumCircuit(qr, cr)

        # Rotation gates for each qubit
        for i in range(self.n_qubits):
            qc.rx(rotation_params[3 * i], qr[i])
            qc.ry(rotation_params[3 * i + 1], qr[i])
            qc.rz(rotation_params[3 * i + 2], qr[i])

        # Entangling CRX gates between neighbouring qubits
        for i in range(self.n_qubits - 1):
            qc.crx(entangle_params[i], qr[i], qr[i + 1])

        qc.measure(qr, cr)
        return qc

    def _patch_to_params(self, patch: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Map a 2×2 patch (4 values) to rotation and entanglement parameters.
        For simplicity, use the pixel values directly as angles (scaled to [-π, π]).
        """
        flat = patch.flatten()
        rotation = np.clip(flat * 2 * np.pi, -np.pi, np.pi)  # 4 angles
        # Duplicate to satisfy 12 rotation parameters (3 per qubit)
        rotation = np.tile(rotation, 3)
        entangle = np.random.uniform(0, 2 * np.pi, self.n_qubits - 1)
        return rotation, entangle

    def run(self, inputs: np.ndarray, shots: int = 1024) -> np.ndarray:
        """
        Args:
            inputs: NumPy array of shape (batch, 1, H, W) with pixel values in [0, 1].
        Returns:
            Weighted patch tensor of shape (batch, 1, H', W') where H' = H // 2, W' = W // 2.
        """
        batch, _, H, W = inputs.shape
        Hp, Wp = H // self.patch_size, W // self.patch_size
        weighted_patches = np.zeros((batch, 1, Hp, Wp))

        for b in range(batch):
            for i in range(Hp):
                for j in range(Wp):
                    # Extract 2×2 patch
                    patch = inputs[b, 0, i*2:i*2+2, j*2:j*2+2]
                    # Map patch to quantum parameters
                    rot_params, ent_params = self._patch_to_params(patch)
                    # Build and execute circuit
                    qc = self._build_circuit(rot_params, ent_params)
                    job = execute(qc, self.backend, shots=shots)
                    result = job.result()
                    counts = result.get_counts(qc)
                    # Convert counts to a probability vector over 2^n outcomes
                    probs = np.array([counts.get(f"{i:0{self.n_qubits}b}", 0) for i in range(2**self.n_qubits)]) / shots
                    # Use the probability of the |00..0> state as an attention weight
                    weight = probs[0]
                    # Weighted patch value
                    weighted_patches[b, 0, i, j] = weight * patch.mean()

        return weighted_patches

__all__ = ["SelfAttention__gen260"]
