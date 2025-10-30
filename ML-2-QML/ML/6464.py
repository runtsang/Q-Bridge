import torch
from torch import nn
import numpy as np

class ClassicalSelfAttention:
    """Classical self‑attention module used in the hybrid model."""
    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        query = torch.as_tensor(inputs @ rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        key = torch.as_tensor(inputs @ entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        value = torch.as_tensor(inputs, dtype=torch.float32)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ value).numpy()

class UnifiedQCNNAttention(nn.Module):
    """Hybrid model combining a classical QCNN backbone and a quantum‑enhanced self‑attention block."""
    def __init__(self,
                 input_dim: int = 8,
                 conv_hidden: int = 16,
                 pool_dim: int = 12,
                 attention_dim: int = 4,
                 use_q_attention: bool = True) -> None:
        super().__init__()
        # Classical QCNN backbone
        self.feature_map = nn.Sequential(
            nn.Linear(input_dim, conv_hidden), nn.Tanh(),
            nn.Linear(conv_hidden, conv_hidden), nn.Tanh()
        )
        self.conv1 = nn.Sequential(nn.Linear(conv_hidden, conv_hidden), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(conv_hidden, pool_dim), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(pool_dim, attention_dim), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(attention_dim, attention_dim), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(attention_dim, attention_dim), nn.Tanh())
        self.head = nn.Linear(attention_dim, 1)

        # Attention modules
        self.classical_attention = ClassicalSelfAttention(embed_dim=attention_dim)
        self.use_q_attention = use_q_attention

        # Parameters for quantum attention (placeholders; in practice these would be optimized on a quantum device)
        self.q_rotation_params = nn.Parameter(torch.randn(attention_dim * 3))
        self.q_entangle_params = nn.Parameter(torch.randn(attention_dim - 1))

        # Parameters for classical attention
        self.c_rotation_params = torch.randn(attention_dim * 3)
        self.c_entangle_params = torch.randn(attention_dim - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Classical QCNN feature extraction
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)

        # Attention step
        if self.use_q_attention:
            # Simulated quantum attention using the classical routine
            rotation = self.q_rotation_params.detach().cpu().numpy()
            entangle = self.q_entangle_params.detach().cpu().numpy()
            x_np = x.detach().cpu().numpy()
            attended = self.classical_attention.run(rotation, entangle, x_np)
            x = torch.from_numpy(attended).to(x.device).float()
        else:
            rotation = self.c_rotation_params
            entangle = self.c_entangle_params
            attended = self.classical_attention.run(rotation, entangle, x.detach().cpu().numpy())
            x = torch.from_numpy(attended).to(x.device).float()

        return torch.sigmoid(self.head(x))

def QCNNAttention() -> UnifiedQCNNAttention:
    """Factory returning a configured UnifiedQCNNAttention instance."""
    return UnifiedQCNNAttention()

__all__ = ["UnifiedQCNNAttention", "QCNNAttention"]
