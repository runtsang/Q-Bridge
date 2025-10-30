import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassicalSelfAttention(nn.Module):
    """Learnable self‑attention block."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        scores = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / (self.embed_dim ** 0.5), dim=-1)
        return torch.matmul(scores, v)

class QuanvolutionFilter(nn.Module):
    """2×2 convolution that flattens to a feature vector."""
    def __init__(self, in_channels: int = 1, out_channels: int = 4, kernel_size: int = 2, stride: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x).view(x.size(0), -1)

class HybridSelfAttentionClassifier(nn.Module):
    """
    Classical hybrid architecture that mirrors the quantum counterpart.
    Attention → Quanvolution → Linear head.
    """
    def __init__(self, embed_dim: int = 4, num_qubits: int = 4, depth: int = 2, num_classes: int = 10):
        super().__init__()
        self.attention = ClassicalSelfAttention(embed_dim)
        self.quanvolution = QuanvolutionFilter()
        self.classifier = nn.Sequential(
            nn.Linear(4 * 14 * 14, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        self.n_qubits = num_qubits
        self.depth = depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attention(x)
        qfeat = self.quanvolution(attn_out)
        logits = self.classifier(qfeat)
        return F.log_softmax(logits, dim=-1)

    def build_quantum_circuit(self):
        """
        Return a Qiskit circuit that reproduces the same topology as the
        classical model: encoding → attention‑style layers → classification ansatz.
        """
        from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
        from qiskit.circuit import ParameterVector
        from qiskit.quantum_info import SparsePauliOp

        qr = QuantumRegister(self.n_qubits, "q")
        cr = ClassicalRegister(self.n_qubits, "c")
        circuit = QuantumCircuit(qr, cr)

        # Encoding layer
        encoding = ParameterVector("x", self.n_qubits)
        for i, param in enumerate(encoding):
            circuit.rx(param, i)

        # Attention‑style block
        attn_params = ParameterVector("theta_attn", self.n_qubits * self.depth)
        idx = 0
        for _ in range(self.depth):
            for q in range(self.n_qubits):
                circuit.ry(attn_params[idx], q)
                idx += 1
            for q in range(self.n_qubits - 1):
                circuit.cz(q, q + 1)

        # Classification ansatz
        cls_params = ParameterVector("theta_cls", self.n_qubits * self.depth)
        idx = 0
        for _ in range(self.depth):
            for q in range(self.n_qubits):
                circuit.ry(cls_params[idx], q)
                idx += 1
            for q in range(self.n_qubits - 1):
                circuit.cz(q, q + 1)

        circuit.measure(qr, cr)
        return circuit, list(encoding), list(attn_params) + list(cls_params), \
               [SparsePauliOp("I" * i + "Z" + "I" * (self.n_qubits - i - 1)) for i in range(self.n_qubits)]

__all__ = ["HybridSelfAttentionClassifier"]
