import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
import numpy as np

class QuantumKernel(tq.QuantumModule):
    """Quantum kernel feature map using a fixed set of single‑qubit Ry rotations."""
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ry"])
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device)
        self.encoder(self.q_device, x)
        out = self.measure(self.q_device)
        return out  # shape (bsz, n_wires)

class ClassicalSelfAttention:
    """Simple dot‑product self‑attention operating on numpy arrays."""
    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        query = torch.as_tensor(inputs @ rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        key = torch.as_tensor(inputs @ entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        value = torch.as_tensor(inputs, dtype=torch.float32)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ value).numpy()

class QFCModel(nn.Module):
    """Classical surrogate of the Quantum‑NAT fully‑connected block."""
    def __init__(self, in_features: int, hidden: int = 64, out_features: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_features),
        )
        self.norm = nn.BatchNorm1d(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.features(x))

class SamplerQNNGen062(nn.Module):
    """Hybrid sampler that combines quantum kernel embedding, self‑attention,
    and a fully‑connected classifier."""
    def __init__(self, input_dim: int = 2, embed_dim: int = 4):
        super().__init__()
        self.quantum_kernel = QuantumKernel(n_wires=embed_dim)
        self.attention = ClassicalSelfAttention(embed_dim=embed_dim)
        self.classifier = QFCModel(in_features=embed_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        embed = self.quantum_kernel(inputs)  # (bsz, embed_dim)
        attn = self.attention.run(embed.numpy(), embed.numpy(), embed.numpy())
        attn = torch.from_numpy(attn).to(inputs.device)
        out = self.classifier(attn)
        return F.softmax(out, dim=-1)

__all__ = ["SamplerQNNGen062"]
