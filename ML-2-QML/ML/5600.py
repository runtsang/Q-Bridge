import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassicalKernelAnsatz(nn.Module):
    """Classical RBF kernel ansatz."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class ClassicalKernel(nn.Module):
    """Wraps the RBF kernel ansatz."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.ansatz = ClassicalKernelAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

class ClassicalQLSTM(nn.Module):
    """Pure PyTorch LSTM tagger."""
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int, tagset_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, tagset_size)

    def tag(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.embedding(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        logits = self.classifier(lstm_out.view(len(sentence), -1))
        return F.log_softmax(logits, dim=1)

class QuanvolutionFilter(nn.Module):
    """Classical convolutional filter inspired by quanvolution."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)

class QCNNModel(nn.Module):
    """Fully‑connected network mimicking a QCNN."""
    def __init__(self):
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

class HybridQuantumKernelModel:
    """Hybrid model that can operate in classical or quantum mode."""
    def __init__(self, mode: str = 'classical', gamma: float = 1.0,
                 embedding_dim: int = 50, hidden_dim: int = 32,
                 vocab_size: int = 10000, tagset_size: int = 10,
                 n_qubits: int = 4):
        self.mode = mode
        if mode == 'classical':
            self.kernel = ClassicalKernel(gamma)
            self.lstm = ClassicalQLSTM(embedding_dim, hidden_dim, vocab_size, tagset_size)
            self.filter = QuanvolutionFilter()
            self.cnn = QCNNModel()
        else:
            # quantum implementations will be defined in qml_code
            self.kernel = None
            self.lstm = None
            self.filter = None
            self.cnn = None

    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor):
        """Return Gram matrix between two batches."""
        if self.mode == 'classical':
            kernel = self.kernel
        else:
            kernel = self.kernel
        return torch.tensor([[kernel(x, y).item() for y in b] for x in a])

    def tag_sequence(self, sentence: torch.Tensor):
        """Run the tagger on a sentence."""
        return self.lstm.tag(sentence)

    def convolve(self, x: torch.Tensor):
        """Apply quanvolutional filter."""
        return self.filter(x)

    def encode(self, x: torch.Tensor):
        """Run through the QCNN‑style network."""
        return self.cnn(x)
