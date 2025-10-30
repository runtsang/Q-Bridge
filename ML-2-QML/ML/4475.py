import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ------------------------------------------------------------------
# Classical QCNN – a fully‑connected emulation of the quantum CNN
# ------------------------------------------------------------------
class QCNNModel(nn.Module):
    """
    Stack of fully connected layers that mimics the depth and
    non‑linearity pattern of the original QCNN.  It accepts a
    feature vector of size 8 and reduces it to a scalar.
    """
    def __init__(self):
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

# ------------------------------------------------------------------
# Classical Fully‑Connected Layer – replaces a quantum FCL
# ------------------------------------------------------------------
class FullyConnectedLayer(nn.Module):
    """
    A single linear layer that emulates the quantum fully‑connected
    layer by returning the mean tanh activation of the weighted
    input parameters.
    """
    def __init__(self, n_features: int = 1):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def run(self, thetas: np.ndarray) -> np.ndarray:
        values = torch.as_tensor(thetas, dtype=torch.float32).view(-1, 1)
        expectation = torch.tanh(self.linear(values)).mean(dim=0)
        return expectation.detach().numpy()

# ------------------------------------------------------------------
# Classical Sampler Network – replaces a quantum SamplerQNN
# ------------------------------------------------------------------
class SamplerModule(nn.Module):
    """
    A small feed‑forward network that outputs a probability
    distribution over two classes.  It is a drop‑in replacement
    for the quantum sampler.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(x), dim=-1)

# ------------------------------------------------------------------
# Hybrid LSTM‑based tagger – classical backbone
# ------------------------------------------------------------------
class QLSTMEnhanced(nn.Module):
    """
    Sequence tagger that can optionally inject quantum sub‑modules.
    The default implementation is fully classical, but flags are
    provided to switch to quantum variants when available.
    """
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 use_quantum_lstm: bool = False,
                 use_quantum_cnn: bool = False,
                 use_quantum_fcl: bool = False,
                 use_quantum_sampler: bool = False):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # Classical LSTM backbone
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

        # Feature extractor – classical QCNN by default
        self.cnn = QCNNModel() if not use_quantum_cnn else None

        # Optional quantum‑style layers (currently placeholders)
        self.fcl = FullyConnectedLayer() if use_quantum_fcl else None
        self.sampler = SamplerModule() if use_quantum_sampler else None

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a batch of sentences.
        ``sentence`` is a LongTensor of shape (seq_len, batch).
        """
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        # Apply QCNN (or placeholder) to each timestep
        if self.cnn is not None:
            cnn_out = self.cnn(lstm_out.view(len(sentence), -1))
        else:
            cnn_out = lstm_out.view(len(sentence), -1)
        logits = self.hidden2tag(cnn_out)

        # Optional quantum‑style post‑processing
        if self.fcl is not None:
            logits = torch.tensor(self.fcl.run(logits.detach().cpu().numpy()))
        if self.sampler is not None:
            logits = self.sampler(logits)

        return F.log_softmax(logits, dim=1)

__all__ = ["QLSTMEnhanced", "QCNNModel", "FullyConnectedLayer", "SamplerModule"]
