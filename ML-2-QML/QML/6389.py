import numpy as np
import qiskit
from qiskit import Aer, execute
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes

class HybridConvModel:
    """Quantum‑enhanced convolutional network that mirrors the classical
    HybridConvModel.  It applies a 2×2 quantum convolution filter followed
    by a variational quantum fully‑connected layer.  The class is a
    drop‑in replacement for the classical model in hybrid experiments.
    """
    def __init__(self, kernel_size: int = 2, n_qubits: int = 4,
                 shots: int = 1024, threshold: float = 0.5,
                 num_classes: int = 4) -> None:
        self.kernel_size = kernel_size
        self.n_qubits = n_qubits
        self.shots = shots
        self.threshold = threshold
        self.num_classes = num_classes

        self.backend = Aer.get_backend("qasm_simulator")

        # 2×2 quantum convolution filter
        self.filter_circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self.filter_circuit.rx(self.theta[i], i)
        self.filter_circuit.barrier()

        # Variational fully‑connected layer
        self.fc_ansatz = RealAmplitudes(self.n_qubits, reps=2)
        self.fc_params = self.fc_ansatz.parameters

    def _run_filter(self, patch: np.ndarray) -> float:
        """Run the quantum convolution filter on a 2×2 patch."""
        patch = patch.reshape(-1)
        bind = {self.theta[i]: np.pi if v > self.threshold else 0
                for i, v in enumerate(patch)}
        circ = self.filter_circuit.bind_parameters(bind)
        job = execute(circ, self.backend, shots=self.shots)
        result = job.result()
        counts = result.get_counts(circ)
        probs = np.array([counts.get(bin(i)[2:].zfill(self.n_qubits), 0)
                          for i in range(2**self.n_qubits)])
        probs /= self.shots
        return probs.sum() / self.n_qubits

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x : np.ndarray
            Input of shape (B, 1, H, W) with values in [0, 1].

        Returns
        -------
        np.ndarray
            Logits of shape (B, num_classes).
        """
        B, C, H, W = x.shape
        conv_out = np.zeros((B, H - self.kernel_size + 1, W - self.kernel_size + 1))
        for b in range(B):
            for i in range(H - self.kernel_size + 1):
                for j in range(W - self.kernel_size + 1):
                    patch = x[b, 0, i:i+self.kernel_size, j:j+self.kernel_size]
                    conv_out[b, i, j] = self._run_filter(patch)

        # Flatten feature map
        flat = conv_out.reshape(B, -1)

        # Variational quantum fully‑connected layer
        logits = np.zeros((B, self.num_classes))
        for b in range(B):
            circ = self.fc_ansatz.copy()
            circ.measure_all()
            job = execute(circ, self.backend, shots=self.shots)
            result = job.result()
            counts = result.get_counts(circ)
            probs = np.array([counts.get(bin(i)[2:].zfill(self.n_qubits), 0)
                              for i in range(2**self.n_qubits)])
            probs /= self.shots
            # Map first num_classes probabilities to logits (placeholder)
            logits[b] = probs[:self.num_classes] * 10

        return logits

__all__ = ["HybridConvModel"]
