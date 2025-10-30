import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pennylane as qml

class SelfAttention(nn.Module):
    """
    Hybrid classical Self‑Attention with optional quantum re‑weighting.
    """
    def __init__(self, embed_dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.query_proj = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.key_proj   = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.value_proj = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.out_proj   = nn.Linear(hidden_dim, embed_dim, bias=False)

        # Quantum device for re‑weighting
        self.dev = qml.device("default.qubit", wires=hidden_dim)
        @qml.qnode(self.dev)
        def circuit(r, e):
            for i in range(hidden_dim):
                qml.RX(r[3 * i], wires=i)
                qml.RY(r[3 * i + 1], wires=i)
                qml.RZ(r[3 * i + 2], wires=i)
            for i in range(hidden_dim - 1):
                qml.CRX(e[i], wires=[i, i + 1])
            return qml.expval(qml.PauliZ(wires=list(range(hidden_dim))))
        self.circuit = circuit

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        Q = self.query_proj(inputs)
        K = self.key_proj(inputs)
        V = self.value_proj(inputs)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(Q.size(-1))
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        return self.out_proj(out)

    def run(self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            inputs: np.ndarray,
            backend: str | None = None) -> np.ndarray:
        """
        run(rotation_params, entangle_params, inputs, backend=None)

        If backend is None, performs pure classical attention.
        If backend is a string identifying a Pennylane device (e.g. "default.qubit"),
        the circuit is executed on that device and its expectation values are used
        as a re‑weighting vector for the attention scores.
        """
        inputs_t = torch.as_tensor(inputs, dtype=torch.float32)
        Q = self.query_proj(inputs_t)
        K = self.key_proj(inputs_t)
        V = self.value_proj(inputs_t)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(Q.size(-1))

        if backend is not None:
            # Use specified quantum backend
            if backend == "default.qubit":
                weights = self.circuit(rotation_params, entangle_params)
            else:
                # Placeholder for other backends
                weights = np.ones(self.hidden_dim)
            weights = torch.tensor(weights, dtype=torch.float32)
            weights = weights.view(1, 1, -1)
            scores = scores * weights

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        return self.out_proj(out).detach().numpy()
