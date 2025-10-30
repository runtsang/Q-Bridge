import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple

class ConvFilter(nn.Module):
    """Classical convolutional filter that emulates the quantum filter."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0, bias: bool = True):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=bias, stride=1)
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # data shape: (batch, 1, H, W)
        out = self.conv(data)
        out = torch.sigmoid(out - self.threshold)
        return out

class QCNNModule(nn.Module):
    """Classical approximation of the QCNN architecture."""
    def __init__(self, in_channels: int = 1, hidden_dim: int = 16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=0),
            nn.Tanh()
        )
        self.pool1 = nn.Sequential(
            nn.Conv2d(hidden_dim, 12, kernel_size=2, stride=2),
            nn.Tanh()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(12, 8, kernel_size=2, stride=1),
            nn.Tanh()
        )
        self.pool2 = nn.Sequential(
            nn.Conv2d(8, 4, kernel_size=2, stride=2),
            nn.Tanh()
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        return x

class ClassicalQLSTM(nn.Module):
    """Purely classical LSTM cell mirroring the quantum interface."""
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
    def forward(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_linear(combined))
            i = torch.sigmoid(self.input_linear(combined))
            g = torch.tanh(self.update_linear(combined))
            o = torch.sigmoid(self.output_linear(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)
    def _init_states(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

class LSTMTagger(nn.Module):
    """Sequence tagging model that can swap between classical and quantum LSTM."""
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int, tagset_size: int, use_quantum: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if use_quantum:
            self.lstm = ClassicalQLSTM(embedding_dim, hidden_dim)  # placeholder for quantum LSTM
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

class HybridModel(nn.Module):
    """End‑to‑end model that chains ConvFilter, QCNNModule and LSTMTagger."""
    def __init__(self, vocab_size: int, tagset_size: int, use_quantum_lstm: bool = False):
        super().__init__()
        self.conv = ConvFilter(kernel_size=2, threshold=0.0)
        self.qcnn = QCNNModule(in_channels=1, hidden_dim=16)
        self.lstm = LSTMTagger(
            embedding_dim=4,  # placeholder; will be set after flattening
            hidden_dim=8,
            vocab_size=vocab_size,
            tagset_size=tagset_size,
            use_quantum=use_quantum_lstm
        )
    def forward(self, images: torch.Tensor, sequence: torch.Tensor) -> torch.Tensor:
        """
        images: (batch, 1, H, W)
        sequence: (seq_len, batch) indices for embedding
        """
        # Feature extraction
        feats = self.conv(images)          # -> (batch, 1, H', W')
        feats = self.qcnn(feats)           # -> (batch, C, H'', W'')
        feats = feats.view(feats.size(0), -1)  # flatten
        # Prepare embeddings for the LSTM
        self.lstm.word_embeddings = nn.Embedding(feats.size(1), self.lstm.word_embeddings.embedding_dim)
        self.lstm.word_embeddings.weight.data = torch.randn(feats.size(1), self.lstm.word_embeddings.embedding_dim)
        # Run sequence tagging
        tags = self.lstm(sequence)
        return tags

def Conv() -> ConvFilter:
    """Factory returning the classical ConvFilter."""
    return ConvFilter(kernel_size=2, threshold=0.0)

__all__ = [
    "ConvFilter",
    "QCNNModule",
    "ClassicalQLSTM",
    "LSTMTagger",
    "HybridModel",
    "Conv",
]
