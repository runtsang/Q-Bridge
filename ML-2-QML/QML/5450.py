"""Hybrid quantum estimator that mirrors the classical HybridEstimator.

The quantum version keeps the same logical data flow:
 1. Classical convolution (2×2 filter with a threshold).
 2. Self‑attention via NumPy soft‑max.
 3. Autoencoder implemented with NumPy linear layers.
 4. Variational quantum circuit (RealAmplitudes) that takes the latent
    representation as input parameters and returns the expectation value
    of a Z observable as the regression output.

The class is fully executable on a Qiskit simulator and can be used
in a hybrid training loop by supplying a classical optimizer for the
quantum parameters.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit.providers.aer import AerSimulator

# ----- Classical preprocessing helpers ------------------------------------
def conv_filter(image: np.ndarray, kernel_size: int = 2, threshold: float = 0.0) -> np.ndarray:
    """Apply a 2‑D convolution with a single kernel and a sigmoid activation."""
    h, w = image.shape
    out_h, out_w = h - kernel_size + 1, w - kernel_size + 1
    out = np.zeros((out_h, out_w), dtype=np.float32)
    for i in range(out_h):
        for j in range(out_w):
            patch = image[i:i+kernel_size, j:j+kernel_size]
            logits = patch.sum()
            out[i, j] = 1.0 / (1.0 + np.exp(-(logits - threshold)))
    return out

def self_attention(inputs: np.ndarray, embed_dim: int = 4) -> np.ndarray:
    """NumPy implementation of a single‑head self‑attention block."""
    # inputs shape (batch, seq_len, embed_dim)
    q = inputs @ np.eye(embed_dim)
    k = inputs @ np.eye(embed_dim)
    v = inputs
    scores = q @ k.transpose(0, 2, 1) / np.sqrt(embed_dim)
    attn = np.exp(scores)
    attn = attn / attn.sum(axis=-1, keepdims=True)
    return attn @ v

class Autoencoder:
    """Simple NumPy autoencoder with linear layers."""
    def __init__(self, input_dim: int, latent_dim: int = 32,
                 hidden_dims: tuple[int, int] = (128, 64)):
        self.w_enc = [np.random.randn(input_dim, hidden_dims[0]),
                      np.random.randn(hidden_dims[0], hidden_dims[1]),
                      np.random.randn(hidden_dims[1], latent_dim)]
        self.b_enc = [np.zeros((hidden_dims[0],)),
                      np.zeros((hidden_dims[1],)),
                      np.zeros((latent_dim,))]
        self.w_dec = [np.random.randn(latent_dim, hidden_dims[1]),
                      np.random.randn(hidden_dims[1], hidden_dims[0]),
                      np.random.randn(hidden_dims[0], input_dim)]
        self.b_dec = [np.zeros((hidden_dims[1],)),
                      np.zeros((hidden_dims[0],)),
                      np.zeros((input_dim,))]

    def encode(self, x: np.ndarray) -> np.ndarray:
        a = x
        for W, b in zip(self.w_enc, self.b_enc):
            a = np.tanh(a @ W + b)
        return a

    def decode(self, z: np.ndarray) -> np.ndarray:
        a = z
        for W, b in zip(self.w_dec, self.b_dec):
            a = np.tanh(a @ W + b)
        return a

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.decode(self.encode(x))

# ----- Hybrid quantum estimator ---------------------------------------------
class HybridEstimator:
    """
    Quantum‑classical hybrid estimator that follows the same logical pipeline
    as the classical HybridEstimator.

    Parameters
    ----------
    latent_dim : int
        Number of qubits used in the variational circuit.
    """

    def __init__(self, latent_dim: int = 32):
        self.latent_dim = latent_dim
        self.backend = AerSimulator()
        self.estimator = StatevectorEstimator(
            backend=self.backend,
            observable=SparsePauliOp.from_list([("Z" * latent_dim, 1)])
        )
        # Classical autoencoder for preprocessing
        self.autoencoder = Autoencoder(input_dim=4, latent_dim=latent_dim)

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Full classical preprocessing pipeline."""
        # Convolution
        conv = conv_filter(image, kernel_size=2, threshold=0.0)
        # Flatten and keep first 4 dims to match embed_dim
        vec = conv.flatten()[:4]
        # Self‑attention (single‑head)
        vec = vec.reshape(1, 1, 4)  # batch, seq_len, embed_dim
        attn_out = self._attention(vec)
        # Autoencoder encoding
        latent = self.autoencoder.encode(attn_out.squeeze(0))
        return latent

    @staticmethod
    def _attention(inputs: np.ndarray) -> np.ndarray:
        """Simple soft‑max attention implemented in NumPy."""
        q = inputs @ np.eye(inputs.shape[-1])
        k = inputs @ np.eye(inputs.shape[-1])
        v = inputs
        scores = q @ k.transpose(0, 2, 1) / np.sqrt(inputs.shape[-1])
        attn = np.exp(scores)
        attn = attn / attn.sum(axis=-1, keepdims=True)
        return attn @ v

    def quantum_circuit(self, latent: np.ndarray) -> QuantumCircuit:
        """Build a parameter‑shiftable circuit that accepts the latent vector."""
        qc = RealAmplitudes(num_qubits=self.latent_dim, reps=1)
        # Embed latent as initial angles
        param_map = {param: val for param, val in zip(qc.parameters, latent)}
        qc = qc.bind_parameters(param_map)
        return qc

    def predict(self, image: np.ndarray) -> float:
        """Run a single prediction."""
        latent = self.preprocess(image)
        qc = self.quantum_circuit(latent)
        # Estimate expectation value of Z observable
        result = self.estimator.run(qc).values[0].real
        return float(result)

__all__ = ["HybridEstimator"]
