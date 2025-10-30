from __future__ import annotations

from typing import Iterable

import torch
from torch import nn

# Core photonic fraud‑detection utilities
from.FraudDetection import FraudLayerParameters, build_fraud_detection_program

# Classical convolution filter
from.Conv import Conv

# Classical transformer for text
from.QTransformerTorch import TextClassifier


class FraudDetectionHybrid(nn.Module):
    """
    Hybrid fraud‑detection model that can run in classical or quantum mode.

    Classical mode:
        * A stack of photonic‑style fully‑connected layers (built by
          :func:`build_fraud_detection_program`).
        * A classical convolution filter (ConvFilter) that extracts a scalar
          feature from a 2‑D kernel.
        * A classical transformer (TextClassifier) for tokenised text.

    Quantum mode:
        * The quantum implementation is provided in the QML module.
    """

    def __init__(
        self,
        mode: str = "classical",
        input_params: FraudLayerParameters | None = None,
        layers: Iterable[FraudLayerParameters] | None = None,
    ) -> None:
        super().__init__()
        self.mode = mode

        # Build the classical fraud‑detection backbone
        self.classical_model = build_fraud_detection_program(
            input_params or FraudLayerParameters(
                bs_theta=0.0,
                bs_phi=0.0,
                phases=(0.0, 0.0),
                squeeze_r=(0.0, 0.0),
                squeeze_phi=(0.0, 0.0),
                displacement_r=(0.0, 0.0),
                displacement_phi=(0.0, 0.0),
                kerr=(0.0, 0.0),
            ),
            layers or [],
        )

        # Linear projection from combined feature vector to the 2‑D input of the
        # photonic model.  The combined vector consists of a scalar convolution
        # output and a 128‑dimensional transformer embedding.
        self.project = nn.Linear(1 + 128, 2)

        # Classical convolution filter
        self.conv_filter = Conv()

        # Classical transformer for text
        self.text_classifier = TextClassifier(
            vocab_size=30522,
            embed_dim=128,
            num_heads=4,
            num_blocks=2,
            ffn_dim=256,
            num_classes=2,
            dropout=0.1,
        )

    def forward_classical(self, image: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the classical branch.

        Parameters
        ----------
        image : torch.Tensor
            2‑D tensor of shape (kernel_size, kernel_size) representing an image patch.
        text : torch.Tensor
            1‑D tensor of token ids.

        Returns
        -------
        torch.Tensor
            Log‑probability of fraud.
        """
        # Convolution feature (scalar)
        conv_out = self.conv_filter.run(image.numpy())

        # Text embedding
        text_out = self.text_classifier(text.unsqueeze(0)).squeeze(0)

        # Combine features and project to the 2‑D input expected by the photonic model
        combined = torch.cat(
            [torch.tensor([conv_out], device=image.device).float(), text_out], dim=0
        )
        projected = self.project(combined.unsqueeze(0))

        # Fraud‑detection forward
        fraud_logit = self.classical_model(projected)
        return fraud_logit.squeeze(0)

    def forward_quantum(self, image: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        """
        Placeholder for quantum forward pass – implemented in the QML module.
        """
        raise NotImplementedError("Quantum forward pass is implemented in the QML module.")

    def forward(self, image: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        if self.mode == "classical":
            return self.forward_classical(image, text)
        else:
            return self.forward_quantum(image, text)


__all__ = ["FraudDetectionHybrid"]
