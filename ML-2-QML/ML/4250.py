import torch
from torch import nn
import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter

class EstimatorQNNGen(nn.Module):
    """
    Hybrid classical‑quantum regressor:
    • Feed‑forward backbone (2→8→4→1)
    • Classical self‑attention with trainable rotation/entangle params
    • 1‑qubit variational circuit providing a quantum feature
    All three outputs are summed to produce the final prediction.
    """
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self.base = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1)
        ).to(self.device)

        # Classical self‑attention
        self.attn = self._build_attention()

        # Quantum circuit
        self.qc = self._build_quantum_circuit()

        # Trainable attention parameters
        self.rotation_params = nn.Parameter(torch.randn(12, device=self.device))
        self.entangle_params = nn.Parameter(torch.randn(3, device=self.device))

    def _build_attention(self):
        class ClassicalSelfAttention:
            def __init__(self, embed_dim: int):
                self.embed_dim = embed_dim

            def run(self, rotation_params: np.ndarray,
                    entangle_params: np.ndarray,
                    inputs: np.ndarray) -> np.ndarray:
                query = torch.as_tensor(
                    inputs @ rotation_params.reshape(self.embed_dim, -1),
                    dtype=torch.float32
                )
                key = torch.as_tensor(
                    inputs @ entangle_params.reshape(self.embed_dim, -1),
                    dtype=torch.float32
                )
                value = torch.as_tensor(inputs, dtype=torch.float32)
                scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
                return (scores @ value).detach().cpu().numpy()
        return ClassicalSelfAttention(embed_dim=4)

    def _build_quantum_circuit(self):
        theta = Parameter("theta")
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.ry(theta, 0)
        qc.measure_all()
        return qc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x).squeeze(-1)  # (batch,)
        # Classical attention
        rotation = self.rotation_params.detach().cpu().numpy()
        entangle = self.entangle_params.detach().cpu().numpy()
        attn_out = self.attn.run(rotation, entangle, base_out.detach().cpu().numpy())
        attn_tensor = torch.as_tensor(attn_out, dtype=torch.float32, device=self.device)

        # Quantum feature
        backend = Aer.get_backend("qasm_simulator")
        bound_qc = self.qc.bind_parameters({ "theta": self.rotation_params.mean().item() })
        job = execute(bound_qc, backend, shots=1024)
        counts = job.result().get_counts(bound_qc)
        probs = np.array(list(counts.values())) / 1024
        probs = probs.astype(float)
        states = np.array(list(counts.keys()), dtype=float)
        expectation = np.sum(states * probs)
        quantum_tensor = torch.tensor([expectation], dtype=torch.float32, device=self.device)

        # Combine streams
        out = base_out + attn_tensor + quantum_tensor.expand_as(base_out)
        return out.unsqueeze(-1)

__all__ = ["EstimatorQNNGen"]
