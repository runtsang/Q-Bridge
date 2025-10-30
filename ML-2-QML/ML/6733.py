import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import itertools
from typing import List, Tuple, Sequence, Iterable

Tensor = torch.Tensor
State = Tuple[Tensor, Tensor]

# Original utilities ----------------------------------------------------
def _random_linear(in_features: int, out_features: int) -> Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    return [(torch.randn(weight.size(1)), weight @ torch.randn(weight.size(1))) for _ in range(samples)]

def random_network(qnn_arch: Sequence[int], samples: int):
    weights = [_random_linear(in_f, out_f) for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:])]
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight

def feedforward(qnn_arch: Sequence[int], weights: Sequence[Tensor], samples: Iterable[Tuple[Tensor, Tensor]]):
    outputs = []
    for x, _ in samples:
        activations = [x]
        current = x
        for w in weights:
            current = torch.tanh(w @ current)
            activations.append(current)
        outputs.append(activations)
    return outputs

def state_fidelity(a: Tensor, b: Tensor) -> float:
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float((a_norm @ b_norm).item() ** 2)

def fidelity_adjacency(states: Sequence[Tensor], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5):
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(s_i, s_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

# Auto‑encoder block ----------------------------------------------------
class AutoEncoderBlock(nn.Module):
    def __init__(self, dim: int, latent_dim: int = 16):
        super().__init__()
        self.encoder = nn.Linear(dim, latent_dim, bias=False)
        self.decoder = nn.Linear(latent_dim, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        z = self.encoder(x)
        return self.decoder(z)

# GraphQNN class --------------------------------------------------------
class GraphQNN(nn.Module):
    def __init__(self, qnn_arch: Sequence[int], autoenc: bool = False, latent_dim: int = 16, device: torch.device | None = None):
        super().__init__()
        self.qnn_arch = list(qnn_arch)
        self.autoenc = autoenc
        self.modules_list = nn.ModuleList()
        self.autoenc_list = nn.ModuleList()
        for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
            self.modules_list.append(nn.Linear(in_f, out_f, bias=False))
            if autoenc:
                self.autoenc_list.append(AutoEncoderBlock(out_f, latent_dim))
            else:
                self.autoenc_list.append(None)
        self.to(device or torch.device('cpu'))

    def forward(self, x: Tensor) -> List[Tensor]:
        activations = [x]
        current = x
        for linear, auto in zip(self.modules_list, self.autoenc_list):
            current = torch.tanh(linear(current))
            activations.append(current)
            if auto is not None:
                recon = auto(current)
                activations.append(recon)
                current = recon
        return activations

    def train_network(self, dataset: List[Tuple[Tensor, Tensor]], lr: float = 1e-3, epochs: int = 100, reg_weight: float = 0.0):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_hist = []
        for _ in range(epochs):
            total_loss = 0.0
            for x, y in dataset:
                optimizer.zero_grad()
                outputs = self.forward(x)
                pred = outputs[-1]
                loss = F.mse_loss(pred, y)
                if reg_weight > 0.0:
                    for i, auto in enumerate(self.autoenc_list):
                        if auto is not None:
                            pre = outputs[2*i+1]
                            post = outputs[2*i+2]
                            loss += reg_weight * F.mse_loss(post, pre)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            loss_hist.append(total_loss / len(dataset))
        return loss_hist

    def inference(self, x: Tensor) -> Tensor:
        return self.forward(x)[-1]

    @staticmethod
    def from_random(qnn_arch: Sequence[int], samples: int, autoenc: bool = False, latent_dim: int = 16):
        arch, weights, training_data, target = random_network(qnn_arch, samples)
        model = GraphQNN(arch, autoenc=autoenc, latent_dim=latent_dim)
        with torch.no_grad():
            for l, w in zip(model.modules_list, weights):
                l.weight.copy_(w.t())
        return model, training_data
