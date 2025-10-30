"""Hybrid classical sampler and quanvolution network.

This module combines a lightweight sampler network with a classical
quanvolution filter.  The sampler produces a probability distribution
over two parameters that are used to weight the input image before
the quanvolution convolution.  The overall architecture is a
drop‑in replacement for the original SamplerQNN and Quanvolution
examples, but it now demonstrates how classical sampling can be
integrated with convolutional feature extraction.

The scaling paradigm is *combination*:  the sampler and the quanvolution
filters are trained jointly, allowing the sampler to learn to
generate feature‑enhancing weights for the convolutional stage.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridSamplerQuanvolution(nn.Module):
    """
    A hybrid classical network that first samples a 2‑dimensional
    probability vector with a small feed‑forward network and then
    feeds the image (augmented with the sampled weights) through a
    quanvolution‑style 2×2 convolution followed by a linear head.
    """

    def __init__(
        self,
        sampler_input_dim: int = 2,
        sampler_hidden_dim: int = 4,
        sampler_output_dim: int = 2,
        quanvolution_channels: int = 4,
        num_classes: int = 10,
    ) -> None:
        super().__init__()

        # Sampler network: 2‑dim input → softmax over 2 outputs
        self.sampler_net = nn.Sequential(
            nn.Linear(sampler_input_dim, sampler_hidden_dim),
            nn.Tanh(),
            nn.Linear(sampler_hidden_dim, sampler_output_dim),
        )

        # Classical quanvolution filter: 2×2 conv with stride 2
        self.quanvolution = nn.Conv2d(
            in_channels=1 + sampler_output_dim,  # image + sampler channels
            out_channels=quanvolution_channels,
            kernel_size=2,
            stride=2,
        )

        # Classifier head
        self.classifier = nn.Linear(
            quanvolution_channels * 14 * 14, num_classes
        )

    def forward(self, x: torch.Tensor, sampler_inputs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input images of shape (batch, 1, 28, 28).
        sampler_inputs : torch.Tensor
            Parameters for the sampler network of shape (batch, 2).

        Returns
        -------
        torch.Tensor
            Log‑softmax logits of shape (batch, num_classes).
        """
        # 1. Sample a probability vector from the sampler network
        probs = F.softmax(self.sampler_net(sampler_inputs), dim=-1)  # (batch, 2)

        # 2. Broadcast the sampler output to match image spatial dims
        # Expand to (batch, 2, 28, 28) then concatenate with image
        sampler_feats = probs.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 28, 28)
        x_aug = torch.cat([x, sampler_feats], dim=1)  # (batch, 1+2, 28, 28)

        # 3. Apply quanvolution convolution
        features = self.quanvolution(x_aug)  # (batch, out_channels, 14, 14)

        # 4. Flatten and classify
        logits = self.classifier(features.view(features.size(0), -1))
        return F.log_softmax(logits, dim=-1)


__all__ = ["HybridSamplerQuanvolution"]
