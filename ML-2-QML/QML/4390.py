from __future__ import annotations

import torch
from torch import nn

# Quantum libraries
import torchquantum as tq
import torchquantum.functional as tqf

# Core components from the seed modules
from.FraudDetection import FraudLayerParameters
from.Conv import Conv
from.QLSTM import QLSTM
from.QTransformerTorch import TextClassifier


class FraudDetectionHybrid(nn.Module):
    """
    Quantum‑enabled fraud‑detection model that mirrors the classical API.

    * A quantum LSTM encoder (:class:`~.QLSTM`) that processes a dummy sequence
      derived from the image patch.
    * A quantum transformer (TextClassifier with ``n_qubits_transformer`` > 0)
      for tokenised text.
    * A quantum convolution filter (QuanvCircuit) that produces a scalar
      feature from a 2‑D kernel.
    * A final linear classifier that aggregates the three modalities.
    """

    def __init__(
        self,
        mode: str = "quantum",
        input_dim: int = 128,
        hidden_dim: int = 128,
        n_qubits_lstm: int = 4,
        n_qubits_transformer: int = 8,
        n_qubits_ffn: int = 8,
        n_qlayers: int = 1,
    ) -> None:
        super().__init__()
        self.mode = mode

        # Quantum LSTM encoder
        self.lstm = QLSTM(input_dim, hidden_dim, n_qubits=n_qubits_lstm)

        # Quantum transformer for text
        self.text_classifier = TextClassifier(
            vocab_size=30522,
            embed_dim=128,
            num_heads=4,
            num_blocks=2,
            ffn_dim=256,
            num_classes=1,
            dropout=0.1,
            n_qubits_transformer=n_qubits_transformer,
            n_qubits_ffn=n_qubits_ffn,
            n_qlayers=n_qlayers,
        )

        # Quantum convolution filter
        self.conv_filter = Conv()

        # Final classifier
        self.classifier = nn.Linear(hidden_dim + 128 + 1, 1)

    def forward_quantum(self, image: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        """
        Quantum forward pass.

        Parameters
        ----------
        image : torch.Tensor
            2‑D tensor of shape (kernel_size, kernel_size).
        text : torch.Tensor
            1‑D tensor of token ids.

        Returns
        -------
        torch.Tensor
            Log‑probability of fraud.
        """
        # Convolution feature (scalar)
        conv_out = self.conv_filter.run(image.numpy())

        # Text transformer
        text_out = self.text_classifier(text.unsqueeze(0)).squeeze(0)

        # Dummy sequence derived from the image for the quantum LSTM
        seq = torch.randn(10, image.size(0), 128, device=image.device)
        lstm_out, _ = self.lstm(seq)
        lstm_out = lstm_out.mean(dim=0)  # aggregate over time

        # Combine modalities
        combined = torch.cat(
            [lstm_out, text_out, torch.tensor([conv_out], device=image.device).float()],
            dim=0,
        )

        # Classification
        return self.classifier(combined.unsqueeze(0)).squeeze(0)

    def forward_classical(self, image: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        """
        Placeholder for classical forward pass – implemented in the ML module.
        """
        raise NotImplementedError("Classical forward pass is implemented in the ML module.")

    def forward(self, image: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        if self.mode == "quantum":
            return self.forward_quantum(image, text)
        else:
            return self.forward_classical(image, text)


__all__ = ["FraudDetectionHybrid"]
