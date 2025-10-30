import numpy as np
import torch
from torch import nn

class Encoder(nn.Module):
    """Dense encoder mapping raw features to a lower dimensional embedding."""
    def __init__(self, input_dim: int, embed_dim: int, hidden_sizes=(64, 32)):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, embed_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class AutoencoderNet(nn.Module):
    """Auto‑encoder that refines embeddings."""
    def __init__(self, embed_dim: int, latent_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

class RBFKernel(nn.Module):
    """Trainable RBF kernel."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * (diff * diff).sum(dim=-1, keepdim=True))

class HybridKernel(nn.Module):
    """Hybrid kernel combining classical RBF and a quantum kernel."""
    def __init__(self,
                 input_dim: int,
                 embed_dim: int,
                 n_qubits: int = 4,
                 quantum_depth: int = 2,
                 use_quantum: bool = True):
        super().__init__()
        self.encoder = Encoder(input_dim, embed_dim)
        self.autoencoder = AutoencoderNet(embed_dim)
        self.rbf = RBFKernel()
        self.use_quantum = use_quantum
        if use_quantum:
            # Lazy import to avoid heavy dependencies when not needed
            from.QuantumKernelMethod__gen400_qml import QuantumKernelPennylane
            self.quantum_kernel = QuantumKernelPennylane(n_qubits=n_qubits,
                                                        depth=quantum_depth)
        else:
            self.quantum_kernel = None

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Encode and auto‑encode
        x_emb = self.autoencoder(self.encoder(x))
        y_emb = self.autoencoder(self.encoder(y))
        # Classical kernel
        rbf_val = self.rbf(x_emb, y_emb).squeeze(-1)
        if self.use_quantum:
            q_val = self.quantum_kernel(x_emb, y_emb).squeeze(-1)
            return 0.5 * rbf_val + 0.5 * q_val
        return rbf_val

def kernel_matrix(a: torch.Tensor,
                  b: torch.Tensor,
                  hybrid: HybridKernel) -> np.ndarray:
    """Compute Gram matrix between two sets of samples using the hybrid kernel."""
    mat = torch.empty((len(a), len(b)), dtype=torch.float32, device=a.device)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            mat[i, j] = hybrid(x, y)
    return mat.cpu().numpy()

__all__ = ["Encoder", "AutoencoderNet", "RBFKernel", "HybridKernel", "kernel_matrix"]
