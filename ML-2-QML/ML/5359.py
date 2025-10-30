import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RBFKernel(nn.Module):
    """Classical RBF kernel used in hybrid pipelines."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

    def matrix(self, a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
        return np.array([[self.forward(x, y).item() for y in b] for x in a])

class ClassicalSelfAttention:
    """Pure‑Python self‑attention block."""
    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim

    def run(self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            inputs: np.ndarray) -> np.ndarray:
        query = torch.as_tensor(inputs @ rotation_params.reshape(self.embed_dim, -1),
                                dtype=torch.float32)
        key = torch.as_tensor(inputs @ entangle_params.reshape(self.embed_dim, -1),
                              dtype=torch.float32)
        value = torch.as_tensor(inputs, dtype=torch.float32)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ value).numpy()

class QuantumQLSTM(nn.Module):
    """Placeholder for a quantum‑enhanced LSTM in the classical module."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim)

    def forward(self, inputs: torch.Tensor, states=None):
        return self.lstm(inputs, states)

class QuantumSelfAttention:
    """Fallback quantum‑attention placeholder that forwards to classical."""
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits

    def run(self, rotation_params, entangle_params, inputs):
        return ClassicalSelfAttention(self.n_qubits).run(rotation_params,
                                                        entangle_params,
                                                        inputs)

class HybridSamplerQNN(nn.Module):
    """Hybrid sampler that unifies classical and quantum components."""
    def __init__(self,
                 embed_dim: int = 4,
                 kernel_gamma: float = 1.0,
                 lstm_hidden: int = 8,
                 lstm_n_qubits: int = 0,
                 attention_n_qubits: int = 0):
        super().__init__()
        # Classical sampler
        self.sampler = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2)
        )
        # Kernel
        self.kernel = RBFKernel(gamma=kernel_gamma)
        # LSTM
        if lstm_n_qubits > 0:
            self.lstm = QuantumQLSTM(embed_dim, lstm_hidden, lstm_n_qubits)
        else:
            self.lstm = nn.LSTM(embed_dim, lstm_hidden)
        # Self‑attention
        if attention_n_qubits > 0:
            self.attention = QuantumSelfAttention(attention_n_qubits)
        else:
            self.attention = ClassicalSelfAttention(embed_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return probability distribution produced by the sampler."""
        return F.softmax(self.sampler(inputs), dim=-1)

    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
        """Compute Gram matrix using the RBF kernel."""
        return self.kernel.matrix(a, b)

    def lstm_forward(self, seq: torch.Tensor) -> torch.Tensor:
        """Run the (classical or quantum) LSTM on a sequence."""
        outputs, _ = self.lstm(seq)
        return outputs

    def attention_forward(self,
                          inputs: np.ndarray,
                          rotation_params: np.ndarray,
                          entangle_params: np.ndarray) -> np.ndarray:
        """Apply self‑attention (classical or quantum) to inputs."""
        return self.attention.run(rotation_params, entangle_params, inputs)

__all__ = ["HybridSamplerQNN", "RBFKernel", "ClassicalSelfAttention",
           "QuantumQLSTM", "QuantumSelfAttention"]
