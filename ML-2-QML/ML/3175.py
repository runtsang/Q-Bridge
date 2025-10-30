import torch
from torch import nn
import torch.nn.functional as F

class UnifiedQCNN(nn.Module):
    """
    Hybrid quantum‑classical convolutional network that unifies
    the dense feature‑map backbone from the QCNN seed with a
    recursive quantum convolution/pooling stack.  The network
    supports a classical head or an optional quantum expectation
    head that can be swapped during training.
    """
    def __init__(self,
                 input_dim: int = 8,
                 feature_layers: int = 3,
                 conv_layers: int = 3,
                 use_quantum_head: bool = False,
                 qnn=None,
                 num_classes: int = 2):
        super().__init__()

        # Feature‑map backbone (classical)
        layers = []
        in_features = input_dim
        for _ in range(feature_layers):
            layers.append(nn.Linear(in_features, 16))
            layers.append(nn.Tanh())
            in_features = 16
        self.feature_map = nn.Sequential(*layers)

        # Classical convolution / pooling blocks
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        for _ in range(conv_layers):
            self.conv_layers.append(nn.Linear(in_features, in_features))
            self.pool_layers.append(nn.Linear(in_features, max(2, in_features // 2)))
            in_features = max(2, in_features // 2)

        # Final head
        self.classifier = nn.Linear(in_features, num_classes)

        # Quantum expectation head
        self.use_quantum_head = use_quantum_head
        self.qnn = qnn

        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature mapping
        x = self.feature_map(x)

        # Convolution & pooling
        for conv, pool in zip(self.conv_layers, self.pool_layers):
            x = F.relu(conv(x))
            x = pool(x)

        # Classical head
        logits = self.classifier(x)

        # Quantum expectation head (overrides classical logits)
        if self.use_quantum_head and self.qnn is not None:
            # The qnn expects a 2‑D tensor of shape (batch, input_dim)
            # and returns a 1‑D tensor of expectations.
            q_expect = self.qnn(x).view(-1, 1)
            # For a binary problem, the quantum expectation serves as the logit
            if self.num_classes == 2:
                logits = q_expect
            else:
                # For multi‑class, prepend the quantum expectation as the first logit
                zeros = torch.zeros_like(q_expect)
                logits = torch.cat([q_expect, zeros], dim=-1)

        # Convert to probabilities
        if self.num_classes == 2:
            probs = torch.sigmoid(logits)
            return torch.cat([probs, 1 - probs], dim=-1)
        else:
            return F.softmax(logits, dim=-1)

__all__ = ["UnifiedQCNN"]
