import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class ParamShiftActivation(torch.autograd.Function):
    """Sigmoid activation with a learnable shift."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float) -> torch.Tensor:
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (outputs,) = ctx.saved_tensors
        grad_inputs = grad_output * outputs * (1 - outputs)
        return grad_inputs, None

class HybridClassifier(nn.Module):
    """Classical CNN + transformer head + param‑shifted sigmoid for binary classification."""
    def __init__(self, in_channels: int = 3, shift: float = 0.0) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.transformer = TransformerEncoder(
            TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=256, dropout=0.1),
            num_layers=2
        )
        self.fc = nn.Linear(64, 1)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)  # (batch, 64)
        # transformer expects seq_len x batch x d_model; add dummy seq dimension
        x = x.unsqueeze(0)
        x = self.transformer(x)
        x = x.squeeze(0)
        logits = self.fc(x)
        probs = ParamShiftActivation.apply(logits, self.shift)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["HybridClassifier", "ParamShiftActivation"]
