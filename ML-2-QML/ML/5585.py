from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HybridClassifier(nn.Module):
    """
    Classical hybrid classifier that combines a feed‑forward network,
    a classical LSTM, a fully‑connected layer, and a quanvolutional
    front‑end.  The architecture is modular and can be partially disabled
    via the *config* dictionary.
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 lstm_layers: int = 1,
                 fcl_units: int = 32,
                 quanv_channels: int = 4,
                 config: dict | None = None) -> None:
        super().__init__()
        self.config = config or {
            "classifier": True,
            "lstm": True,
            "fcl": True,
            "quanv": True,
        }

        if self.config["quanv"]:
            self.quanv = nn.Conv2d(1, quanv_channels, kernel_size=2, stride=2)
        else:
            self.quanv = nn.Identity()

        if self.config["fcl"]:
            self.fcl = nn.Linear(fcl_units, 1)
        else:
            self.fcl = nn.Identity()

        if self.config["lstm"]:
            self.lstm = nn.LSTM(input_dim, hidden_dim, lstm_layers, batch_first=True)
        else:
            self.lstm = nn.Identity()

        if self.config["classifier"]:
            layers = [nn.Linear(input_dim, 64), nn.ReLU(),
                      nn.Linear(64, 32), nn.ReLU(),
                      nn.Linear(32, 2)]
            self.classifier = nn.Sequential(*layers)
        else:
            self.classifier = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4 and self.config["quanv"]:
            x = self.quanv(x)
            x = x.view(x.size(0), -1)

        if self.config["fcl"]:
            x = self.fcl(x)

        if self.config["lstm"] and x.dim() == 2:
            x = x.unsqueeze(1)

        if self.config["lstm"]:
            x, _ = self.lstm(x)

        logits = self.classifier(x)
        return F.log_softmax(logits, dim=-1)

__all__ = ["HybridClassifier"]
