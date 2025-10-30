"""Classical hybrid network merging convolution, graph fidelity, and a small regressor."""  
import torch  
import torch.nn as nn  
import torch.nn.functional as F  
import networkx as nx  

class QuanvolutionHybrid(nn.Module):  
    """  
    Classical hybrid architecture inspired by Quanvolution, GraphQNN, and EstimatorQNN.  
    The network applies a 2×2 convolution, aggregates patch features via a fidelity graph,  
    and performs regression with a lightweight feed‑forward head.  
    """  

    def __init__(  
        self,  
        num_filters: int = 4,  
        kernel_size: int = 2,  
        stride: int = 2,  
        hidden_dim: int = 128,  
        output_dim: int = 10,  
        fidelity_threshold: float = 0.8,  
    ) -> None:  
        super().__init__()  
        self.conv = nn.Conv2d(1, num_filters, kernel_size=kernel_size, stride=stride)  
        self.hidden = nn.Linear(num_filters * 14 * 14, hidden_dim)  
        self.regressor = nn.Linear(hidden_dim, output_dim)  
        self.fidelity_threshold = fidelity_threshold  

    def _extract_patches(self, x: torch.Tensor) -> torch.Tensor:  
        """Return a tensor of shape [B, N, F] where N=14×14 and F=num_filters."""  
        patches = self.conv(x)  # [B, F, H, W]  
        B, F, H, W = patches.shape  
        return patches.permute(0, 2, 3, 1).reshape(B, H * W, F)  

    def _aggregate_fidelity(self, patches: torch.Tensor) -> torch.Tensor:  
        """Aggregate patch features using a thresholded fidelity graph."""  
        B, N, F = patches.shape  
        patches_norm = patches / patches.norm(dim=2, keepdim=True).clamp_min(1e-7)  
        sims = torch.bmm(patches_norm, patches_norm.transpose(1, 2))  
        adj = (sims >= self.fidelity_threshold).float()  
        deg = adj.sum(dim=2, keepdim=True).clamp_min(1e-7)  
        agg = torch.bmm(adj, patches) / deg  
        return agg.reshape(B, -1)  

    def fidelity_graph(self, patches: torch.Tensor, threshold: float | None = None) -> nx.Graph:  
        """Return a NetworkX graph built from patch fidelities (for inspection)."""  
        if threshold is None:  
            threshold = self.fidelity_threshold  
        patches_norm = patches / patches.norm(dim=1, keepdim=True).clamp_min(1e-7)  
        sims = torch.mm(patches_norm, patches_norm.t())  
        G = nx.Graph()  
        G.add_nodes_from(range(patches.size(0)))  
        for i in range(patches.size(0)):  
            for j in range(i + 1, patches.size(0)):  
                if sims[i, j] >= threshold:  
                    G.add_edge(i, j, weight=1.0)  
        return G  

    def forward(self, x: torch.Tensor) -> torch.Tensor:  
        patches = self._extract_patches(x)  
        agg = self._aggregate_fidelity(patches)  
        h = F.relu(self.hidden(agg))  
        out = self.regressor(h)  
        return F.log_softmax(out, dim=-1)  

__all__ = ["QuanvolutionHybrid"]
