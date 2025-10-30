"""Combined classical regression model incorporating convolutional and fully‑connected layers."""  

from __future__ import annotations  

import numpy as np  
import torch  
from torch import nn  
from torch.utils.data import Dataset  

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:  
    """Generate synthetic regression data with a sinusoidal target."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)  
    angles = x.sum(axis=1)  
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)  
    return x, y.astype(np.float32)  

class RegressionDataset(Dataset):  
    """Simple PyTorch dataset that mirrors the quantum data format."""  

    def __init__(self, samples: int, num_features: int):  
        self.features, self.labels = generate_superposition_data(num_features, samples)  

    def __len__(self) -> int:  # type: ignore[override]  
        return len(self.features)  

    def __getitem__(self, index: int):  # type: ignore[override]  
        return {  
            "states": torch.tensor(self.features[index], dtype=torch.float32),  
            "target": torch.tensor(self.labels[index], dtype=torch.float32),  
        }  

# ---- Classical auxiliary layers ----  

def ConvFilter(kernel_size: int = 2, threshold: float = 0.0) -> nn.Module:  
    """1‑D convolutional filter that emulates a quantum quanvolution layer."""  

    class _ConvFilter(nn.Module):  
        def __init__(self):  
            super().__init__()  
            self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, bias=True)  
            self.threshold = threshold  

        def forward(self, x: torch.Tensor) -> torch.Tensor:  
            # x: (batch, features) -> (batch, 1, features)  
            x = x.unsqueeze(1)  
            logits = self.conv(x)  # (batch, 1, L-k+1)  
            activations = torch.sigmoid(logits - self.threshold)  
            return activations.mean(dim=-1).squeeze(-1)  # (batch,)  

    return _ConvFilter()  

def FullyConnectedLayer(n_features: int = 1) -> nn.Module:  
    """Linear layer that mimics a quantum fully‑connected block."""  

    class _FullyConnectedLayer(nn.Module):  
        def __init__(self):  
            super().__init__()  
            self.linear = nn.Linear(n_features, 1)  

        def forward(self, x: torch.Tensor) -> torch.Tensor:  
            return torch.tanh(self.linear(x)).mean(dim=0, keepdim=True)  

    return _FullyConnectedLayer()  

# ---- Combined classical model ----  

class CombinedModel(nn.Module):  
    """Regression model that chains a convolution, a fully‑connected layer, and a linear head."""  

    def __init__(self, num_features: int, conv_kernel: int = 2, hidden_dim: int = 8):  
        super().__init__()  
        self.conv = ConvFilter(kernel_size=conv_kernel)  
        self.fc = FullyConnectedLayer(n_features=1)  
        self.head = nn.Linear(1, 1)  

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  
        # state_batch: (batch, num_features)  
        conv_out = self.conv(state_batch)  # (batch,)  
        conv_out = conv_out.unsqueeze(-1)  # (batch, 1)  
        fc_out = self.fc(conv_out)  # (batch, 1)  
        return self.head(fc_out).squeeze(-1)  # (batch,)  

__all__ = ["CombinedModel", "RegressionDataset", "generate_superposition_data"]
