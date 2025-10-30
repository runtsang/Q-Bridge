import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class HybridClassifier(nn.Module):
    '''Deep MLP baseline for binary classification.

    The network replaces the quantum expectation head with a
    multi‑layer perceptron that offers comparable expressivity
    while being lightweight and fully classical.
    '''
    def __init__(self, in_features: int, hidden_layers: Tuple[int,...] = (256, 128, 64)):
        super().__init__()
        layers = []
        prev = in_features
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.mlp(x)
        probs = torch.sigmoid(logits)
        return torch.cat([probs, 1 - probs], dim=-1)

    def evaluate(self, dataloader, device='cpu'):
        '''Compute ROC‑AUC and calibration curve on a dataset.'''
        from sklearn.metrics import roc_auc_score, brier_score_loss
        self.eval()
        probs, labels = [], []
        with torch.no_grad():
            for batch, target in dataloader:
                batch = batch.to(device)
                target = target.to(device)
                out = self(batch)
                probs.append(out[:, 0].cpu())
                labels.append(target.cpu())
        probs = torch.cat(probs).numpy()
        labels = torch.cat(labels).numpy()
        auc = roc_auc_score(labels, probs)
        calib = brier_score_loss(labels, probs)
        return {'auc': auc, 'brier': calib}

__all__ = ['HybridClassifier']
