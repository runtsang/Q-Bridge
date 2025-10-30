import torch
import torch.nn as nn
import torch.nn.functional as F
from.qml import QuantumPatchFilter, HybridQuantumHead

class QCNetExtended(nn.Module):
    """
    Hybrid CNN that applies a quantum patch filter to each 2×2 patch of the input image,
    then passes the resulting feature vector through a small classical head and finally
    through a quantum expectation head.  The architecture combines ideas from the
    classical CNN, the quanvolution filter, the fraud‑detection clipping logic and the
    quantum hybrid head.
    """
    def __init__(self,
                 patch_size: int = 2,
                 threshold: float = 0.0,
                 n_qubits: int = 4,
                 shots: int = 100,
                 fc_hidden: int = 120):
        super().__init__()
        self.patch_size = patch_size
        self.threshold = threshold
        self.quantum_patch_filter = QuantumPatchFilter(patch_size, threshold, shots)
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        # 28×28 image → 14×14 patches of size 2×2 → 196 patches
        self.patch_feature_dim = 196
        self.fc1 = nn.Linear(self.patch_feature_dim, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, 84)
        self.fc3 = nn.Linear(84, 1)
        self.hybrid_head = HybridQuantumHead(n_qubits, shots)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply quantum patch filter to the raw image
        patch_features = self.quantum_patch_filter.run_batch(x)
        # Classical fully‑connected head
        x = F.relu(self.fc1(patch_features))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # Quantum expectation head
        probs = self.hybrid_head(x.squeeze(-1))
        return torch.cat((probs, 1 - probs), dim=-1)
