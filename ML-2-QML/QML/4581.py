import numpy as np
import qiskit
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit import Parameter

__all__ = ["ConvGenHybrid"]

class QuantumAttention:
    """Variational attention circuit that returns a tensor of shape (B, S, 1)."""
    def __init__(self, backend=None, shots=1024):
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots

    def run(self, x: np.ndarray) -> np.ndarray:
        # x shape: (B, S, embed_dim)
        B, S, E = x.shape
        out = np.zeros((B, S, 1))
        for b in range(B):
            for s in range(S):
                theta = float(x[b, s, 0])  # use first feature as rotation angle
                qc = QuantumCircuit(1, 1)
                qc.ry(theta, 0)
                qc.measure(0, 0)
                job = execute(qc, self.backend, shots=self.shots)
                counts = job.result().get_counts(qc)
                p0 = counts.get('0', 0) / self.shots
                p1 = counts.get('1', 0) / self.shots
                exp_z = p0 - p1
                out[b, s, 0] = exp_z
        return out

class QuantumFeedForward:
    """Variational feed‑forward circuit that outputs a tensor of shape (B, S, 1)."""
    def __init__(self, backend=None, shots=1024):
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots

    def run(self, x: np.ndarray) -> np.ndarray:
        B, S, E = x.shape
        out = np.zeros((B, S, 1))
        for b in range(B):
            for s in range(S):
                theta1 = float(x[b, s, 0]) if E > 0 else 0.0
                theta2 = float(x[b, s, 1]) if E > 1 else 0.0
                qc = QuantumCircuit(2, 2)
                qc.ry(theta1, 0)
                qc.ry(theta2, 1)
                qc.cx(0, 1)
                qc.measure(0, 0)
                qc.measure(1, 1)
                job = execute(qc, self.backend, shots=self.shots)
                counts = job.result().get_counts(qc)
                exp_z = 0.0
                for bitstring, cnt in counts.items():
                    bit = int(bitstring[1])  # measurement order is [q1, q0]
                    exp_z += (1 if bit == 0 else -1) * cnt
                exp_z /= self.shots
                out[b, s, 0] = exp_z
        return out

class ConvGenHybrid:
    """Hybrid model that uses quantum circuits for attention and feed‑forward."""
    def __init__(self,
                 embed_dim=32,
                 num_heads=4,
                 ffn_dim=64,
                 num_blocks=2,
                 kernel_size=2,
                 threshold=0.0,
                 num_features=1,
                 use_quantum_attention=True,
                 use_quantum_ffn=True,
                 dropout=0.1):
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.attn = QuantumAttention()
        self.ffn = QuantumFeedForward()
        self.classifier = lambda x: np.mean(x, axis=1, keepdims=True)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        # inputs shape (B, 1, H, W)
        B, C, H, W = inputs.shape
        seq = H * W
        x = inputs.reshape(B, seq, C)
        # dummy fully connected step
        thetas = np.arange(seq, dtype=np.float32)
        fcl_out = np.tanh(np.mean(thetas))
        x = x + fcl_out
        # quantum attention
        attn_out = self.attn.run(x)
        # quantum feed‑forward
        ffn_out = self.ffn.run(attn_out)
        # pooling and classification
        pooled = np.mean(ffn_out, axis=1)
        logits = self.classifier(pooled)
        return logits
