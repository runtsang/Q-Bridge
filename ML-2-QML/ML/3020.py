import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SamplerQNNHybrid(nn.Module):
    """Hybrid classical sampler network inspired by SamplerQNN and Quantum‑NAT."""
    def __init__(self, num_classes: int = 4, seed: int | None = None) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        self.norm = nn.BatchNorm1d(num_classes)
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        feat = self.features(x)
        flat = feat.view(bsz, -1)
        logits = self.fc(flat)
        probs = self.norm(logits)
        probs = F.softmax(probs, dim=-1)
        return probs

    def sample(self, probs: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """Draw discrete samples from the probability distribution per batch element."""
        probs_np = probs.detach().cpu().numpy()
        batch = probs_np.shape[0]
        samples = np.zeros((batch, num_samples, probs_np.shape[1]), dtype=int)
        for i in range(batch):
            samples[i] = np.random.choice(
                probs_np.shape[1], size=num_samples, p=probs_np[i]
            )
        return torch.tensor(samples, device=probs.device, dtype=torch.long)

def SamplerQNN():
    return SamplerQNNHybrid()

__all__ = ["SamplerQNNHybrid", "SamplerQNN"]
