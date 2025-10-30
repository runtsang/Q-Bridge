import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridQuantumNat(nn.Module):
    """Classical hybrid model that combines a CNN encoder, a sampler network,
    a QCNN-style fully connected stack, and an optional LSTM tagger."""

    def __init__(self,
                 input_channels: int = 1,
                 num_classes: int = 4,
                 vocab_size: int = 1000,
                 tagset_size: int = 10):
        super().__init__()
        # CNN encoder (from QuantumNAT)
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Fully connected projection
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )
        self.norm = nn.BatchNorm1d(num_classes)

        # Sampler network (mirror of SamplerQNN)
        self.sampler = nn.Sequential(
            nn.Linear(num_classes, 4),
            nn.Tanh(),
            nn.Linear(4, num_classes),
        )

        # QCNN-style fully connected stack
        self.qcnn_stack = nn.Sequential(
            nn.Linear(8, 16), nn.Tanh(),
            nn.Linear(16, 16), nn.Tanh(),
            nn.Linear(16, 12), nn.Tanh(),
            nn.Linear(12, 8), nn.Tanh(),
            nn.Linear(8, 4), nn.Tanh(),
            nn.Linear(4, 1),
        )

        # LSTM tagger
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, batch_first=True)
        self.tag_head = nn.Linear(64, tagset_size)

    def forward(self,
                images: torch.Tensor,
                seq: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass.

        Args:
            images: Tensor of shape (B, C, H, W).
            seq: Optional sequence tensor of shape (B, T, 128) for tagging.

        Returns:
            If seq is None: Tensor of shape (B, 1) representing QCNN output.
            If seq is provided: Tuple[Tensor, Tensor] where first element is QCNN output
            and second element is tag logits of shape (B, T, tagset_size).
        """
        x = self.encoder(images)
        flat = x.view(x.size(0), -1)
        logits = self.fc(flat)
        normed = self.norm(logits)
        sampled = F.softmax(self.sampler(normed), dim=-1)
        qc_out = torch.sigmoid(self.qcnn_stack(sampled))

        if seq is not None:
            lstm_out, _ = self.lstm(seq)
            tag_logits = self.tag_head(lstm_out)
            return qc_out, tag_logits
        return qc_out
