import torch
import torch.nn as nn

class QuantumHybridClassifier(nn.Module):
    'Classical head for the hybrid quantum classifier.'
    def __init__(self, in_features: int, shift: float = 0.0, trainable_shift: bool = False) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        if trainable_shift:
            self.shift = nn.Parameter(torch.tensor(shift, dtype=torch.float32))
        else:
            self.shift = shift
        self.trainable_shift = trainable_shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)
        probs = torch.sigmoid(logits + self.shift)
        return torch.cat([probs, 1 - probs], dim=-1)

    def reset_params(self) -> None:
        'Reinitialize linear weights and bias, and reset shift to zero.'
        self.linear.reset_parameters()
        if self.trainable_shift:
            self.shift.data.fill_(0.0)

__all__ = ['QuantumHybridClassifier']
