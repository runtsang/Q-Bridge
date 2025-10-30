"""QuantumNATEnhanced: classical CNN backbone with multi‑task head."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumNATEnhanced(nn.Module):
    """
    Classical hybrid model that combines a 2‑layer CNN with a fully‑connected
    head capable of producing both a 4‑class classification output and a
    single‑value regression output.  The module is intentionally flexible
    so it can be trained with either a cross‑entropy loss (classification),
    a mean‑squared‑error loss (regression), or a weighted sum of both.
    """
    def __init__(self, num_classes: int = 4, regression: bool = True):
        super().__init__()
        # Backbone: 2‑layer CNN
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # Fully‑connected heads
        self.classifier = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )
        if regression:
            self.regressor = nn.Sequential(
                nn.Linear(16 * 7 * 7, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 1),
            )
        self.norm = nn.BatchNorm1d(num_classes + (1 if regression else 0))
        self.regression = regression

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Tensor of shape (batch, 1, 28, 28)

        Returns:
            classification logits of shape (batch, num_classes)
            regression output of shape (batch, 1) if regression is True
        """
        bsz = x.shape[0]
        features = self.features(x)
        flattened = features.view(bsz, -1)
        logits = self.classifier(flattened)
        out = logits
        if self.regression:
            reg = self.regressor(flattened)
            out = torch.cat([logits, reg], dim=1)
        out = self.norm(out)
        if self.regression:
            return out[:, :4], out[:, 4:]
        return out

    def compute_loss(self, logits, targets, loss_fn_cls, loss_fn_reg=None,
                     weight_cls: float = 1.0, weight_reg: float = 1.0):
        """
        Compute a weighted sum of classification and regression losses.
        """
        loss = loss_fn_cls(logits, targets)
        if self.regression and loss_fn_reg is not None:
            loss += weight_reg * loss_fn_reg(logits[:, 4:], targets[:, 4:])
        return loss

__all__ = ["QuantumNATEnhanced"]
