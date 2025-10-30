import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridEstimatorQNN(nn.Module):
    """
    Classical hybrid estimator with three operating modes.

    * ``regression`` – a shallow fully‑connected network.
    * ``sequence``   – a conventional LSTM tagger.
    * ``image``      – a 2‑D convolution followed by a linear head.

    The class is intentionally lightweight so it can be swapped with the quantum
    counterpart without changing downstream code.
    """

    def __init__(self,
                 mode: str = "regression",
                 input_dim: int = 2,
                 hidden_dim: int = 8,
                 vocab_size: int = 1000,
                 tagset_size: int = 10,
                 num_classes: int = 10) -> None:
        super().__init__()
        self.mode = mode.lower()

        if self.mode == "regression":
            self.regression_net = nn.Sequential(
                nn.Linear(input_dim, 8),
                nn.Tanh(),
                nn.Linear(8, 4),
                nn.Tanh(),
                nn.Linear(4, 1),
            )

        elif self.mode == "sequence":
            self.embedding = nn.Embedding(vocab_size, 50)
            self.lstm = nn.LSTM(50, hidden_dim, batch_first=True)
            self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

        elif self.mode == "image":
            self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)
            self.fc = nn.Linear(4 * 14 * 14, num_classes)

        else:
            raise ValueError(f"Unknown mode {mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "regression":
            return self.regression_net(x)

        if self.mode == "sequence":
            embeds = self.embedding(x)
            lstm_out, _ = self.lstm(embeds)
            logits = self.hidden2tag(lstm_out)
            return F.log_softmax(logits, dim=-1)

        if self.mode == "image":
            features = self.conv(x)
            features = features.view(features.size(0), -1)
            logits = self.fc(features)
            return F.log_softmax(logits, dim=-1)

        raise RuntimeError("unreachable")

__all__ = ["HybridEstimatorQNN"]
