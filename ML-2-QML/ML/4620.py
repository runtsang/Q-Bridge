import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np

# --------------------------------------------------------------------------- #
#  Classical data generator
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Produces a dataset of real‑valued features and a continuous target.
    The target is a sinusoidal function of the sum of all features with a
    small cosine noise term.  The function is identical to the one used in
    the original Quantum‑Regression seed but is expanded to support
    *all* feature sizes.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

# --------------------------------------------------------------------------- #
#  Dataset wrapper
# --------------------------------------------------------------------------- #
class RegressionDataset(Dataset):
    """
    Wraps the data returned by ``generate_superposition...`` for use with a
    DataLoader.  The ``states`` field is kept for compatibility with the
    Quantum‑NAT style API where a quantum circuit expects a batch of states.
    """
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {"states": torch.tensor(self.features[index], dtype=torch.float32),
                "target": torch.tensor(self.labels[index], dtype=torch.float32)}

# --------------------------------------------------------------------------- #
#  Classical sub‑networks
# --------------------------------------------------------------------------- #
class ClassicalRegressionHead(nn.Module):
    """
    Fully‑connected head that maps any input feature vector to a scalar
    regression output.  Inspired by the simple FC block of the
    Quantum‑NAT example.
    """
    def __init__(self, in_features: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

class ClassicalImageClassifier(nn.Module):
    """
    CNN backbone followed by a 4‑dimensional projection, mirroring
    the QFCModel from the Quantum‑NAT seed.  The output can be further
    processed by a linear classifier for any downstream task.
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # 4‑dim embedding
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        feats = self.features(x)
        flat = feats.view(bsz, -1)
        out = self.fc(flat)
        return self.norm(out)

class ClassicalLSTMTagger(nn.Module):
    """
    Sequence tagging model that uses a standard LSTM and a linear head.
    """
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int, tagset_size: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=-1)

# --------------------------------------------------------------------------- #
#  Hybrid model switching between classical modes
# --------------------------------------------------------------------------- #
class QuantumFusionModel(nn.Module):
    """
    A unified interface that can operate in one of three classical modes:
        *'regression'   – simple fully‑connected regression head
        * 'image'        – CNN + 4‑dim embedding
        *'sequence'     – LSTM tagging
    The class is intentionally lightweight and can be easily extended to
    incorporate quantum modules (see the corresponding QML implementation).
    """
    def __init__(self,
                 mode: str,
                 num_features: int = 10,
                 embedding_dim: int = 50,
                 hidden_dim: int = 64,
                 vocab_size: int = 1000,
                 tagset_size: int = 10,
                 in_channels: int = 1):
        super().__init__()
        assert mode in {'regression', 'image','sequence'}, f"Unknown mode {mode}"
        self.mode = mode
        if mode =='regression':
            self.model = ClassicalRegressionHead(num_features)
        elif mode == 'image':
            self.model = ClassicalImageClassifier(in_channels, num_classes=tagset_size)
        else:  # sequence
            self.model = ClassicalLSTMTagger(embedding_dim, hidden_dim, vocab_size, tagset_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

__all__ = ["QuantumFusionModel", "RegressionDataset", "generate_superposition_data"]
