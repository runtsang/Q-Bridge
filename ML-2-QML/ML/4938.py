import torch
import torch.nn as nn
import numpy as np

class QuantumNATModel(nn.Module):
    """
    Classical hybrid model that combines a CNN, a self‑attention block,
    and a regression / classification head.
    """
    def __init__(self, num_classes: int = 4, embed_dim: int = 4):
        super().__init__()
        # Convolutional feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Self‑attention helper
        self.attn_module = self._build_attention(embed_dim)
        # Fully connected head
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7 + embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )
        self.norm = nn.BatchNorm1d(num_classes)

    def _build_attention(self, embed_dim: int):
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
                return (scores @ value).numpy()
        return ClassicalSelfAttention(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        feats = self.features(x)
        flat = feats.view(bsz, -1)
        # Random attention parameters for demonstration
        rot_params = np.random.randn(self.attn_module.embed_dim, 4)
        ent_params = np.random.randn(self.attn_module.embed_dim, 4)
        attn_out = self.attn_module.run(rot_params, ent_params, flat.cpu().numpy())
        attn_tensor = torch.from_numpy(attn_out).to(x.device)
        concat = torch.cat([flat, attn_tensor], dim=1)
        out = self.fc(concat)
        return self.norm(out)

# ---------------------------------------------------------------------------

def generate_superposition_data(num_features: int, samples: int):
    """
    Generate a synthetic regression dataset based on superposition states.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(torch.utils.data.Dataset):
    """
    PyTorch dataset wrapper for the synthetic regression data.
    """
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

def EstimatorQNN():
    """
    Small fully‑connected regression network that mirrors the Qiskit EstimatorQNN example.
    """
    class EstimatorNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 8),
                nn.Tanh(),
                nn.Linear(8, 4),
                nn.Tanh(),
                nn.Linear(4, 1),
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.net(inputs)

    return EstimatorNN()
