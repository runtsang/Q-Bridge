import torch
import numpy as np
import qiskit
from qiskit import execute, Aer
from qiskit.circuit.random import random_circuit

class ClassicalConvFilter(torch.nn.Module):
    """Standard 2‑D convolution with a learnable bias and a sigmoid threshold."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = torch.nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.conv(x)
        return torch.sigmoid(logits - self.threshold)

class QuantumConvFilter:
    """A low‑depth Qiskit circuit that encodes a kernel‑size patch into a probability."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0, shots: int = 200):
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.shots = shots
        self.backend = Aer.get_backend("qasm_simulator")
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> qiskit.QuantumCircuit:
        qc = qiskit.QuantumCircuit(self.n_qubits)
        theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i, p in enumerate(theta):
            qc.rx(p, i)
        qc.barrier()
        qc += random_circuit(self.n_qubits, 2)
        qc.measure_all()
        return qc

    def run(self, data: np.ndarray) -> float:
        """Return the average probability of measuring |1> across all qubits."""
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for row in data:
            bind = {qiskit.circuit.Parameter(f"theta{i}"): (np.pi if val > self.threshold else 0)
                    for i, val in enumerate(row)}
            param_binds.append(bind)
        job = execute(self.circuit, self.backend, shots=self.shots, parameter_binds=param_binds)
        result = job.result()
        counts = result.get_counts(self.circuit)
        total = self.shots * self.n_qubits
        ones = sum(bitstring.count('1') * freq for bitstring, freq in counts.items())
        return ones / total

class RBFKernel:
    """Classical RBF kernel used for patch similarity in the hybrid mode."""
    def __init__(self, gamma: float = 1.0):
        self.gamma = gamma

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        diff = x - y
        return np.exp(-self.gamma * np.sum(diff * diff))

class HybridConvFilter(torch.nn.Module):
    """Patch‑wise quantum kernel followed by a 1×1 classical convolution."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0, gamma: float = 1.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.quantum_filter = QuantumConvFilter(kernel_size, threshold)
        self.rbf = RBFKernel(gamma)
        self.conv = torch.nn.Conv2d(1, 1, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        pad = self.kernel_size // 2
        padded = torch.nn.functional.pad(x, (pad, pad, pad, pad), mode='reflect')
        patches = padded.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1)
        # patches: (B, C, H', W', k, k)
        patches = patches.contiguous().view(B, C, H, W, -1)
        sims = torch.zeros(B, H, W, device=x.device)
        for i in range(H):
            for j in range(W):
                patch = patches[:, :, i, j, :].detach().cpu().numpy()
                sims[:, i, j] = torch.tensor([self.quantum_filter.run(p) for p in patch])
        sims = sims.unsqueeze(1)  # (B,1,H,W)
        return self.conv(sims)

class ConvGen(torch.nn.Module):
    """
    Unified convolutional layer that can operate in classical, quantum, or hybrid mode.
    """
    def __init__(self, kernel_size: int = 2, conv_type: str = "hybrid",
                 threshold: float = 0.0, gamma: float = 1.0):
        super().__init__()
        self.conv_type = conv_type.lower()
        if self.conv_type == "classical":
            self.filter = ClassicalConvFilter(kernel_size, threshold)
        elif self.conv_type == "quantum":
            self.filter = QuantumConvFilter(kernel_size, threshold)
        else:  # hybrid
            self.filter = HybridConvFilter(kernel_size, threshold, gamma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.conv_type == "quantum":
            # quantum filter returns a scalar per image; broadcast to match spatial dims
            B = x.shape[0]
            out = torch.zeros(B, 1, x.shape[2] - self.filter.n_qubits + 1,
                              x.shape[3] - self.filter.n_qubits + 1, device=x.device)
            for b in range(B):
                out[b, 0] = self.filter.run(x[b, 0].detach().cpu().numpy())
            return out
        else:
            return self.filter(x)

__all__ = ["ConvGen", "ClassicalConvFilter", "QuantumConvFilter", "HybridConvFilter"]
