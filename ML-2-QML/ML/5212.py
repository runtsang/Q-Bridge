import torch
from torch import nn
from dataclasses import dataclass
from typing import Iterable, Sequence, List
import torch.nn.functional as F

@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    weight = torch.tensor(
        [
            [params.bs_theta, params.bs_phi],
            [params.squeeze_r[0], params.squeeze_r[1]],
        ],
        dtype=torch.float32,
    )
    bias = torch.tensor(params.phases, dtype=torch.float32)
    if clip:
        weight = weight.clamp(-5.0, 5.0)
        bias = bias.clamp(-5.0, 5.0)
    linear = nn.Linear(2, 2)
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)
    activation = nn.Tanh()
    scale = torch.tensor(params.displacement_r, dtype=torch.float32)
    shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()

class FraudDetectionHybrid(nn.Module):
    """
    Hybrid fraud‑detection model that combines:
    * Photonic‑inspired fully connected layers
    * A 2‑D quanvolution filter (Conv2d) for patch‑wise feature extraction
    * Graph‑based aggregation of intermediate layer outputs using cosine similarity
    * A final linear classifier producing a fraud probability
    """
    def __init__(
        self,
        layer_params: List[FraudLayerParameters],
        conv_out_channels: int = 4,
        graph_threshold: float = 0.8,
    ) -> None:
        super().__init__()
        if not layer_params:
            raise ValueError("At least one layer parameter set must be provided.")
        self.graph_threshold = graph_threshold

        # Photonic‑style layers
        self.layers = nn.ModuleList(
            [_layer_from_params(p, clip=(i!= 0)) for i, p in enumerate(layer_params)]
        )

        # Quanvolution filter
        self.qfilter = nn.Conv2d(1, conv_out_channels, kernel_size=2, stride=2)

        # Linear head after graph aggregation
        self.classifier = nn.Linear(conv_out_channels * 14 * 14, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 2). Two raw fraud features.
        Returns
        -------
        torch.Tensor
            Fraud probability logits of shape (batch, 1).
        """
        # Stage 1: photonic layers
        activations = []
        out = x
        for layer in self.layers:
            out = layer(out)
            activations.append(out)

        # Stage 2: graph aggregation
        # Compute cosine similarity between first two activations
        sims = F.cosine_similarity(activations[0], activations[1], dim=-1)
        weight = torch.where(sims >= self.graph_threshold, torch.ones_like(sims), torch.zeros_like(sims))
        graph_output = activations[0] * weight.unsqueeze(-1) + activations[1] * (1 - weight).unsqueeze(-1)

        # Stage 3: reshape to 2D image for quanvolution
        img = torch.zeros((x.size(0), 1, 28, 28), device=x.device)
        img[:, 0, 0, 0] = graph_output[:, 0]
        img[:, 0, 0, 1] = graph_output[:, 1]

        # Apply quanvolution
        qfeat = self.qfilter(img)
        qfeat = qfeat.view(x.size(0), -1)

        # Final classifier
        logits = self.classifier(qfeat)
        return logits

__all__ = ["FraudLayerParameters", "FraudDetectionHybrid"]
