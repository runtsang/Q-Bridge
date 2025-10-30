import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

class SamplerQNN(nn.Module):
    """
    Classical sampler network that maps a 2D input to a probability distribution
    over 2 outputs. The network has a hidden layer with ReLU activation,
    dropout for regularization, and a final softmax output. It is fully trainable
    with cross‑entropy loss and an Adam optimizer.
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 8, output_dim: int = 2, dropout_rate: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(x), dim=-1)

    def train_step(self, optimizer: torch.optim.Optimizer, loss_fn, inputs: torch.Tensor, targets: torch.Tensor):
        """
        Performs a single training step: forward pass, loss computation,
        backward pass, and optimizer step.
        """
        optimizer.zero_grad()
        outputs = self.forward(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        return loss.item()
