from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np
from typing import Sequence, Iterable, Callable, List

# Lightweight estimator for batches of inputs and observables
class FastBaseEstimator:
    """Evaluate neural network outputs for batches of inputs and observables."""
    def __init__(self, model: nn.Module):
        self.model = model

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        self.model.eval()
        results: List[List[float]] = []
        with torch.no_grad():
            for params in parameter_sets:
                # each param set is a list of floats (e.g., thresholds)
                inputs = torch.tensor(params, dtype=torch.float32).unsqueeze(0)
                outputs = self.model(inputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().cpu())
                    else:
                        scalar = float(val)
                    row.append(scalar)
                results.append(row)
        return results

class GenConv096(nn.Module):
    """
    Classical hybrid convolution filter with optional LSTM gating and
    graph‑based feature adjacency.  The interface is intentionally
    compatible with the original Conv.py seed.
    """
    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        use_lstm: bool = False,
        use_graph: bool = False,
        graph_threshold: float = 0.8,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

        self.use_lstm = use_lstm
        if use_lstm:
            seq_len = kernel_size * kernel_size
            self.lstm = nn.LSTM(
                input_size=seq_len, hidden_size=seq_len, batch_first=True
            )

        self.use_graph = use_graph
        self.graph_threshold = graph_threshold

    def forward(self, x: torch.Tensor, threshold: float | None = None) -> float:
        """
        Args:
            x: input tensor of shape (batch, 1, H, W)
            threshold: optional override for conv activation threshold
        Returns:
            mean activation after optional gating and graph aggregation
        """
        if threshold is None:
            threshold = self.threshold
        conv_out = self.conv(x)
        activations = torch.sigmoid(conv_out - threshold)

        if self.use_lstm:
            seq = activations.view(activations.size(0), -1).unsqueeze(1)
            lstm_out, _ = self.lstm(seq)
            activations = lstm_out.squeeze(1).view_as(activations)

        if self.use_graph:
            states = activations.view(activations.size(0), -1)
            graph = nx.Graph()
            graph.add_nodes_from(range(states.size(0)))
            for i in range(states.size(0)):
                for j in range(i + 1, states.size(0)):
                    fid = torch.dot(states[i], states[j]) / (
                        torch.norm(states[i]) * torch.norm(states[j]) + 1e-12
                    )
                    if fid >= self.graph_threshold:
                        graph.add_edge(i, j, weight=1.0)
            aggregated = torch.zeros_like(activations)
            for comp in nx.connected_components(graph):
                idx = list(comp)
                aggregated[idx] = activations[idx].mean(dim=0, keepdim=True)
            activations = aggregated

        return activations.mean().item()

    def evaluate_thresholds(
        self, image: torch.Tensor, thresholds: Sequence[float]
    ) -> List[float]:
        """
        Evaluate the filter for a fixed image over a list of threshold values
        using FastBaseEstimator.
        """
        class ThresholdModel(nn.Module):
            def __init__(self, conv, img):
                super().__init__()
                self.conv = conv
                self.img = img

            def forward(self, thresh_vals: torch.Tensor) -> torch.Tensor:
                out = []
                for t in thresh_vals:
                    out.append(self.conv(self.img, t))
                return torch.tensor(out)

        model = ThresholdModel(self, image)
        estimator = FastBaseEstimator(model)
        results = estimator.evaluate(
            [lambda o: o], [[t] for t in thresholds]
        )
        return [r[0] for r in results]
