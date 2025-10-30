import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.circuit import Parameter
from qiskit.providers.aer import AerSimulator
import torch
from torch import nn
from typing import Iterable, Sequence, Optional

# --------------------------------------------------------------------------- #
# 1. Quantum fully‑connected layer (derived from FCL.py)
# --------------------------------------------------------------------------- #
class QuantumFCL:
    """
    Simple parameterized circuit that returns the expectation value of a
    single qubit after a Ry rotation.  This mimics the classical FCL.run()
    method but uses a quantum backend.
    """
    def __init__(self, n_qubits: int = 1, shots: int = 1024):
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = AerSimulator()
        self.theta = Parameter('theta')
        self.circuit = QuantumCircuit(n_qubits)
        self.circuit.h(0)
        self.circuit.ry(self.theta, 0)
        self.circuit.barrier()
        self.circuit.measure_all()

    def run(self, theta: float) -> np.ndarray:
        job = execute(self.circuit, self.backend,
                      parameter_binds=[{self.theta: theta}],
                      shots=self.shots)
        result = job.result()
        counts = result.get_counts()
        # Convert to expectation value of Z (0 -> +1, 1 -> -1)
        exp_val = 0.0
        for bitstring, cnt in counts.items():
            exp_val += (1 if bitstring == '0' else -1) * cnt
        exp_val /= self.shots
        return np.array([exp_val])

# --------------------------------------------------------------------------- #
# 2. Quantum kernel (derived from QuantumKernelMethod.py)
# --------------------------------------------------------------------------- #
class QuantumKernel:
    """
    Implements a simple SWAP‑test based kernel that estimates the fidelity
    between two input vectors encoded as rotation angles on separate qubits.
    """
    def __init__(self, n_qubits: int = 1, shots: int = 1024):
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = AerSimulator()

    def swap_test(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        qc = QuantumCircuit(3)
        qc.h(0)
        for i, val in enumerate(x):
            qc.ry(val, 1)
        for i, val in enumerate(y):
            qc.ry(val, 2)
        qc.cx(1, 2)
        qc.cx(2, 1)
        qc.h(0)
        qc.measure_all()
        job = execute(qc, self.backend, shots=self.shots)
        result = job.result()
        counts = result.get_counts()
        # Probability of ancilla being 0
        p0 = sum(cnt for bit, cnt in counts.items() if bit[-1] == '0') / self.shots
        # Fidelity approximation
        return np.array([2 * p0 - 1])

    def run(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.swap_test(x, y)

# --------------------------------------------------------------------------- #
# 3. Quantum transformer block (derived from QTransformerTorch.py)
# --------------------------------------------------------------------------- #
class QuantumTransformerBlock:
    """
    Very light‑weight quantum attention block.  Each token is encoded
    into a small quantum circuit, the expectation values of Pauli‑Z
    are extracted, and a classical linear layer produces the final
    token representation.  The block is wrapped in a standard PyTorch
    module for compatibility.
    """
    def __init__(self,
                 embed_dim: int,
                 n_qubits: int = 4,
                 shots: int = 512):
        self.embed_dim = embed_dim
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = AerSimulator()
        self.theta = Parameter('theta')
        self.circuit = QuantumCircuit(n_qubits)
        for i in range(n_qubits):
            self.circuit.ry(self.theta, i)
        self.circuit.measure_all()
        # Classical linear layers to combine the quantum features
        self.linear1 = nn.Linear(n_qubits, embed_dim)
        self.linear2 = nn.Linear(embed_dim, embed_dim)

    def _quantum_features(self, token: torch.Tensor) -> torch.Tensor:
        # Encode token into the circuit
        qc = self.circuit.copy()
        for i, val in enumerate(token.tolist()[:self.n_qubits]):
            qc.ry(val, i)
        job = execute(qc, self.backend, shots=self.shots)
        result = job.result()
        counts = result.get_counts()
        # Expectation of Pauli‑Z on each qubit
        exp = []
        for i in range(self.n_qubits):
            exp_val = 0.0
            for bitstring, cnt in counts.items():
                bit = bitstring[self.n_qubits - 1 - i]
                exp_val += (1 if bit == '0' else -1) * cnt
            exp_val /= self.shots
            exp.append(exp_val)
        return torch.tensor(exp, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, embed_dim) – only the first n_qubits are used
        to feed the quantum circuit; the rest are ignored for simplicity.
        """
        batch, seq_len, _ = x.shape
        out = torch.zeros((batch, seq_len, self.embed_dim))
        for b in range(batch):
            for s in range(seq_len):
                qfeat = self._quantum_features(x[b, s])
                out[b, s] = self.linear2(torch.relu(self.linear1(qfeat)))
        return out

# --------------------------------------------------------------------------- #
# 4. Unified hybrid model
# --------------------------------------------------------------------------- #
class UnifiedHybridLayer:
    """
    Quantum‑centric implementation of the UnifiedHybridLayer.  The API
    mirrors the classical version but uses Qiskit circuits for the
    FCL, kernel, and transformer components.
    """
    def __init__(self,
                 n_features: int = 1,
                 gamma: float = 1.0,
                 embed_dim: int = 32,
                 num_heads: int = 4,
                 ffn_dim: int = 64,
                 num_blocks: int = 2,
                 num_classes: int = 2,
                 shots: int = 512):
        self.fcl = QuantumFCL(n_qubits=1, shots=shots)
        self.kernel = QuantumKernel(n_qubits=1, shots=shots)
        self.transformer = nn.Sequential(
            *[QuantumTransformerBlock(embed_dim, n_qubits=4, shots=shots)
              for _ in range(num_blocks)]
        )
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, thetas: Iterable[float], x: torch.Tensor) -> torch.Tensor:
        # 1. Run the quantum FCL
        fcl_out = self.fcl.run(thetas[0])  # assume a single theta for simplicity
        # 2. Kernel: similarity between fcl_out and itself
        kernel_out = self.kernel.run(fcl_out, fcl_out)
        # 3. Build a sequence for the transformer: replicate kernel_out
        seq_len = x.size(1)
        seq = torch.tensor(kernel_out, dtype=torch.float32).repeat(seq_len, 1).unsqueeze(0)
        # 4. Transformer
        transformed = self.transformer(seq)
        # 5. Classification
        pooled = transformed.mean(dim=1)
        return self.classifier(pooled)
