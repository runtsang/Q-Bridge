"""Extended Quanvolution model with dual‑branch fusion and contrastive alignment.

The original classical model applies a 2×2 convolution in one‑channel images. This version adds a lightweight “quantum‑style” branch implemented as a learnable linear map on 2×2 patches, and a learnable gate that weighs the two branches. A contrastive loss is provided to encourage the branches to learn compatible embeddings.

Usage:
    model = QuanvolutionFusion()
    logits = model(x)          # forward pass
    loss = model.contrastive_loss(cls_feats, q_feats)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuanvolutionFusion(nn.Module):
    """Hybrid classical‑quantum fusion for MNIST‑style data.

    The model expects a single‑channel image of shape ``(B, 1, 28, 28)``.
    It processes the image with two parallel branches:

    * ``classical_branch`` – a shallow 2×2 convolution followed by a
      flattening operation (exactly the same as the original seed).
    * ``quantum_branch``   – a lightweight “quantum‑style” feature extractor
      implemented as a learnable linear map that operates on 2×2 patches.
      This mimics the measurement statistics of a 4‑qubit circuit but
      stays purely classical so that the module can be trained on a CPU.

    A learnable gate ``alpha`` (initialised to 0.5) controls the
    contribution of each branch.  The final feature vector is a weighted
    sum of the two branches.

    The module also exposes a static method :meth:`contrastive_loss`
    that can be used during training to encourage the two branches to
    produce similar embeddings.
    """

    def __init__(self) -> None:
        super().__init__()
        # Classical branch
        self.classical_conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)
        # Quantum‑style branch: linear mapping on 2×2 patches
        self.quantum_linear = nn.Linear(4, 4)
        # Gate parameter (0 ≤ alpha ≤ 1)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        # Final classifier
        self.classifier = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Classical branch
        cls_feat = self.classical_conv(x)      # shape (B, 4, 14, 14)
        cls_flat = cls_feat.view(x.size(0), -1)  # (B, 784)

        # Quantum‑style branch
        # Extract 2×2 patches and flatten each to length 4
        patches = x.unfold(2, 2, 2).unfold(3, 2, 2)  # (B, 1, 14, 14, 2, 2)
        patches = patches.contiguous().view(x.size(0), 14, 14, 4)
        qfeat = self.quantum_linear(patches)  # (B, 14, 14, 4)
        qfeat = qfeat.view(x.size(0), -1)     # (B, 784)

        # Fuse with learnable gate
        alpha = torch.sigmoid(self.alpha)      # keep between 0 and 1
        feat = alpha * cls_flat + (1 - alpha) * qfeat

        logits = self.classifier(feat)
        return F.log_softmax(logits, dim=-1)

    @staticmethod
    def contrastive_loss(cls_feats: torch.Tensor,
                         q_feats: torch.Tensor,
                         temperature: float = 0.5
                         ) -> torch.Tensor:
        """Compute a simple NT-Xent loss between two feature sets.

        Args:
            cls_feats: Features from the classical branch, shape (B, D).
            q_feats:   Features from the quantum‑style branch, shape (B, D).
            temperature: Temperature scaling for the cosine similarity.

        Returns:
            Scalar contrastive loss encouraging the two branches to
            produce similar embeddings.
        """
        # Normalize features
        cls_norm = F.normalize(cls_feats, dim=1)
        q_norm = F.normalize(q_feats, dim=1)

        # Compute cosine similarity matrix (B, B)
        sim = cls_norm @ q_norm.t() / temperature

        # Labels: diagonal entries are positives
        labels = torch.arange(cls_feats.size(0), device=cls_feats.device)

        loss = F.cross_entropy(sim, labels)
        return loss
