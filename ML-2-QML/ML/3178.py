import torch
from torch import nn
import networkx as nx
import itertools

class ConvGraphQNN:
    """Hybrid classical convolution + graph neural network.

    This class combines a 2‑D convolutional filter with a graph‑based
    feed‑forward network.  It is a drop‑in replacement for the
    original ``Conv`` and ``GraphQNN`` modules but exposes a single
    ``forward`` method that returns a feature vector.

    Parameters
    ----------
    conv_kernel_size : int, optional
        Size of the square convolution kernel.
    conv_threshold : float, optional
        Threshold used in the sigmoid activation of the conv output.
    graph_threshold : float, optional
        Fidelity threshold for constructing graph edges.
    graph_secondary : float, optional
        Lower threshold that still creates an edge with a reduced weight.
    graph_secondary_weight : float, optional
        Weight of edges created by the secondary threshold.
    """

    def __init__(self,
                 conv_kernel_size: int = 2,
                 conv_threshold: float = 0.0,
                 graph_threshold: float = 0.8,
                 graph_secondary: float | None = None,
                 graph_secondary_weight: float = 0.5):
        self.conv_kernel_size = conv_kernel_size
        self.conv_threshold = conv_threshold
        self.graph_threshold = graph_threshold
        self.graph_secondary = graph_secondary
        self.graph_secondary_weight = graph_secondary_weight

        # 1‑channel -> 1‑channel conv (bias=True) replicates the original
        self.conv = nn.Conv2d(1, 1, kernel_size=conv_kernel_size, bias=True)

    # ------------------------------------------------------------------
    # Convolution helper
    # ------------------------------------------------------------------
    def _conv_forward(self, image: torch.Tensor) -> torch.Tensor:
        """Apply the 2‑D convolution and return a flattened feature vector."""
        # Expected image shape: (H, W)
        image = image.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        conv_out = self.conv(image)              # (1,1,H',W')
        conv_out = conv_out.squeeze()            # (H',W')
        # Apply sigmoid with threshold
        conv_out = torch.sigmoid(conv_out - self.conv_threshold)
        return conv_out.flatten()

    # ------------------------------------------------------------------
    # Graph helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
        """Squared cosine similarity between two 1‑D tensors."""
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float((a_norm @ b_norm).item() ** 2)

    def _build_fidelity_graph(self, features: torch.Tensor) -> nx.Graph:
        """Create a weighted adjacency graph from feature fidelities."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(features)))
        for (i, fi), (j, fj) in itertools.combinations(enumerate(features), 2):
            fid = self._state_fidelity(fi, fj)
            if fid >= self.graph_threshold:
                graph.add_edge(i, j, weight=1.0)
            elif self.graph_secondary is not None and fid >= self.graph_secondary:
                graph.add_edge(i, j, weight=self.graph_secondary_weight)
        return graph

    # ------------------------------------------------------------------
    # Feed‑forward helper
    # ------------------------------------------------------------------
    def feedforward(self,
                    weights: list[torch.Tensor],
                    inputs: torch.Tensor) -> torch.Tensor:
        """Simple tanh‑activated feed‑forward network."""
        h = inputs
        for w in weights:
            h = torch.tanh(w @ h)
        return h

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def forward(self,
                image: torch.Tensor,
                weights: list[torch.Tensor] | None = None) -> torch.Tensor:
        """Process an image through the hybrid architecture.

        Parameters
        ----------
        image : torch.Tensor
            2‑D tensor of shape (H, W).
        weights : list[torch.Tensor], optional
            List of weight matrices for the optional feed‑forward stage.
            If ``None`` the raw convolution features are returned.

        Returns
        -------
        torch.Tensor
            Either the raw convolution features or the output of the
            feed‑forward network.
        """
        features = self._conv_forward(image)
        self.graph = self._build_fidelity_graph(features)
        if weights is None:
            return features
        return self.feedforward(weights, features)
