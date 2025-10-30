import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HybridSelfAttentionModel(nn.Module):
    """
    Hybrid classical‑quantum self‑attention model.
    Combines a CNN feature extractor (inspired by Quantum‑NAT),
    a classical multi‑head self‑attention block, and an optional
    quantum attention sub‑module built with Qiskit.
    """

    def __init__(self,
                 embed_dim: int = 64,
                 n_qubits: int = 4,
                 use_quantum: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_qubits = n_qubits
        self.use_quantum = use_quantum

        # CNN feature extractor (Quantum‑NAT style)
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Projection to embed_dim
        self.proj = nn.Linear(16 * 7 * 7, embed_dim)

        # Classical self‑attention parameters
        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_proj   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Quantum attention parameters (trainable)
        if self.use_quantum:
            # Each qubit needs 3 rotation angles + 1 entanglement angle
            self.rotation_params = nn.Parameter(
                torch.randn(n_qubits * 3))
            self.entangle_params = nn.Parameter(
                torch.randn(n_qubits - 1))
        else:
            self.rotation_params = None
            self.entangle_params = None

        # Output head
        self.out = nn.Linear(embed_dim, 4)
        self.norm = nn.BatchNorm1d(4)

    def _classical_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard scaled dot‑product attention over the feature tokens.
        """
        Q = self.query_proj(x)   # (B, T, D)
        K = self.key_proj(x)
        V = self.value_proj(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.embed_dim)
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, V)

    def _quantum_attention(self, x: torch.Tensor, backend, shots=1024) -> torch.Tensor:
        """
        Delegates to the QML module and returns expectation values
        as a tensor of shape (B, n_qubits).
        """
        from.qml import QuantumSelfAttention
        q_attention = QuantumSelfAttention(n_qubits=self.n_qubits, backend=backend)
        # Broadcast parameters across batch
        rot = self.rotation_params.expand(x.size(0), -1).cpu().numpy()
        ent = self.entangle_params.expand(x.size(0), -1).cpu().numpy()
        # Run circuit for each sample and stack results
        exp_vals = []
        for r, e in zip(rot, ent):
            exp = q_attention.run(r, e, shots=shots)
            exp_vals.append(exp)
        return torch.tensor(np.stack(exp_vals), dtype=torch.float32, device=x.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1, 28, 28) image batch
        """
        bsz = x.size(0)
        feats = self.features(x)
        flat = feats.view(bsz, -1)
        embed = self.proj(flat)  # (B, D)

        # Classical attention
        attn_out = self._classical_attention(embed.unsqueeze(1))  # (B, 1, D)
        attn_out = attn_out.squeeze(1)

        # Quantum attention if enabled
        if self.use_quantum:
            # For simplicity, we use a dummy backend; user can supply any Qiskit backend.
            from qiskit.providers.fake_provider import FakeProvider
            backend = FakeProvider().get_backend('fake_qasm_simulator')
            q_attn = self._quantum_attention(embed, backend)
            # Concatenate classical and quantum features
            out = torch.cat([attn_out, q_attn], dim=-1)
        else:
            out = attn_out

        out = self.out(out)
        out = self.norm(out)
        return out

__all__ = ["HybridSelfAttentionModel"]
