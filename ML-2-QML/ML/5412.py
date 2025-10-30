from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridQCNN(nn.Module):
    """Hybrid QCNN model combining Quanvolution, QCNN, and SamplerQNN ideas."""
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 2,
        depth: int = 3,
        use_quanvolution: bool = True,
        use_sampler: bool = False,
    ) -> None:
        super().__init__()
        self.use_quanvolution = use_quanvolution
        self.use_sampler = use_sampler

        # 1. Quanvolution‑style feature extractor
        if self.use_quanvolution:
            self.feature_extractor = nn.Conv2d(
                input_channels, 4, kernel_size=2, stride=2, bias=False
            )
            self.flatten = nn.Flatten()
        else:
            self.feature_extractor = nn.Identity()
            self.flatten = nn.Identity()

        # 2. Quantum‑inspired conv/pool blocks
        in_dim = 4 * 14 * 14 if self.use_quanvolution else 8
        self.conv_blocks = nn.ModuleList()
        self.pool_blocks = nn.ModuleList()

        for _ in range(depth):
            conv = nn.Sequential(nn.Linear(in_dim, 16), nn.Tanh())
            pool = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
            self.conv_blocks.append(conv)
            self.pool_blocks.append(pool)
            in_dim = 12

        # 3. Final classifier head
        self.classifier = nn.Linear(in_dim, num_classes)

        # 4. Optional sampler head (SamplerQNN style)
        if self.use_sampler:
            self.sampler_head = nn.Sequential(
                nn.Linear(num_classes, 4),
                nn.Tanh(),
                nn.Linear(4, num_classes),
                nn.Softmax(dim=-1),
            )
        else:
            self.sampler_head = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_quanvolution:
            x = self.feature_extractor(x)
            x = self.flatten(x)
        else:
            x = self.feature_extractor(x)

        for conv, pool in zip(self.conv_blocks, self.pool_blocks):
            x = conv(x)
            x = pool(x)

        logits = self.classifier(x)

        if self.sampler_head is not None:
            logits = self.sampler_head(logits)

        return logits
