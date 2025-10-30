import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridQuanvolutionNAT(nn.Module):
    """
    Classical hybrid network that mimics the patch‑wise quantum kernel
    with a random linear projection and a classical CNN backbone.
    """

    def __init__(self) -> None:
        super().__init__()
        # CNN backbone to reduce spatial resolution to 14×14
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Random linear mapping from a 2×2×16 patch to 4 features
        self.patch_transform = nn.Linear(2 * 2 * 16, 4)
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(4 * 14 * 14, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        features = self.cnn(x)          # shape: (bsz, 16, 14, 14)
        patches = []
        # Extract non‑overlapping 2×2 patches
        for r in range(0, features.size(2), 2):
            for c in range(0, features.size(3), 2):
                patch = features[:, r:r + 2, c:c + 2, :].view(bsz, -1)
                patch_feat = self.patch_transform(patch)
                patches.append(patch_feat)
        patch_features = torch.cat(patches, dim=1)   # shape: (bsz, 4*14*14)
        logits = self.classifier(patch_features)
        return F.log_softmax(logits, dim=-1)


__all__ = ["HybridQuanvolutionNAT"]
