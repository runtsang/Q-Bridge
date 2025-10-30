import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
from torch.utils.data import Dataset

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic regression dataset of quantum states.
    Each state is |ψ⟩ = cos(θ)|0…0⟩ + e^{iφ} sin(θ)|1…1⟩.
    The target is y = sin(2θ)·cos(φ).
    """
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = (
            np.cos(thetas[i]) * omega_0
            + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
        )
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels

class RegressionDataset(Dataset):
    """
    Dataset that returns quantum state amplitudes and target scalars.
    """
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QuantumFeatureEncoder(tq.QuantumModule):
    """
    Variational quantum circuit that maps an input amplitude vector into a low‑dimensional feature vector.
    """
    def __init__(self, n_wires: int, feature_dim: int):
        super().__init__()
        self.n_wires = n_wires
        self.feature_dim = feature_dim
        # Encode the amplitude vector into the device
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
        )
        # Trainable rotation gates
        self.param_gates = nn.ModuleList(
            [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
        )
        # Entangling layer
        self.entangle = tqf.cnot
        # Measurement to produce a feature vector
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
        # x: [bsz, 2**n_wires]
        self.encoder(qdev, x)
        for gate, wire in zip(self.param_gates, range(self.n_wires)):
            gate(qdev, wires=wire)
        # Ring entanglement
        for w in range(self.n_wires - 1):
            self.entangle(qdev, wires=[w, w + 1])
        self.entangle(qdev, wires=[self.n_wires - 1, 0])
        return self.measure(qdev)  # [bsz, n_wires]

class SimpleTransformerEncoder(nn.Module):
    """
    Lightweight transformer encoder used in the quantum module.
    """
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x

class UnifiedRegressionTransformer(tq.QuantumModule):
    """
    Hybrid quantum‑classical regression model that extends a transformer encoder with a quantum feature extractor.
    """
    def __init__(
        self,
        num_wires: int,
        embed_dim: int = 32,
        num_heads: int = 4,
        ffn_dim: int = 64,
        q_device: tq.QuantumDevice | None = None,
    ):
        super().__init__()
        self.input_proj = nn.Linear(num_wires, embed_dim)
        self.classical_encoder = SimpleTransformerEncoder(embed_dim, num_heads, ffn_dim)
        self.quantum_encoder = QuantumFeatureEncoder(num_wires, num_wires)
        self.head = nn.Linear(embed_dim + num_wires, 1)
        self.q_device = q_device or tq.QuantumDevice(n_wires=num_wires)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        # Classical path
        token = self.input_proj(state_batch)  # [B, E]
        token = token.unsqueeze(1)  # [B, 1, E]
        token = self.classical_encoder(token)  # [B, 1, E]

        # Quantum path
        qdev = self.q_device.copy(bsz=state_batch.shape[0], device=state_batch.device)
        q_features = self.quantum_encoder(state_batch, qdev)  # [B, n_wires]

        # Concatenate and produce output
        combined = torch.cat([token.squeeze(1), q_features], dim=-1)
        return self.head(combined).squeeze(-1)

__all__ = ["UnifiedRegressionTransformer", "RegressionDataset", "generate_superposition_data"]
