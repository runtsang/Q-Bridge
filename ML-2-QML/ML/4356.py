"""Hybrid graph neural network combining classical and quantum-inspired components.

This module extends the original GraphQNN by integrating:
- A torch-based autoencoder for dimensionality reduction of node embeddings.
- A lightweight sampler neural network to produce probability distributions.
- A QCNN-style fully connected model to emulate quantum convolution/pooling.
- Graph construction from state fidelities to enable graph-based learning.
"""

import itertools
from typing import List, Tuple, Sequence, Iterable

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Classical GraphQNN utilities (from original seed)
# --------------------------------------------------------------------------- #

def _random_linear(in_features: int, out_features: int) -> torch.Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: torch.Tensor, samples: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    dataset: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    weights: List[torch.Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight

def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[torch.Tensor],
    samples: Iterable[Tuple[torch.Tensor, torch.Tensor]],
) -> List[List[torch.Tensor]]:
    stored: List[List[torch.Tensor]] = []
    for features, _ in samples:
        activations = [features]
        current = features
        for weight in weights:
            current = torch.tanh(weight @ current)
            activations.append(current)
        stored.append(activations)
    return stored

def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)

def fidelity_adjacency(
    states: Sequence[torch.Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

# --------------------------------------------------------------------------- #
# Autoencoder utilities (from reference 2)
# --------------------------------------------------------------------------- #

class AutoencoderConfig:
    def __init__(self, input_dim: int, latent_dim: int = 32, hidden_dims: Tuple[int, int] = (128, 64), dropout: float = 0.1):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout

class AutoencoderNet(nn.Module):
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        encoder_layers = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(inputs))

def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
) -> AutoencoderNet:
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    return AutoencoderNet(config)

def _as_tensor(data: torch.Tensor | Iterable[float]) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data
    return torch.as_tensor(data, dtype=torch.float32)

def train_autoencoder(
    model: AutoencoderNet,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> List[float]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = torch.utils.data.TensorDataset(_as_tensor(data))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: List[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction = model(batch)
            loss = loss_fn(reconstruction, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

# --------------------------------------------------------------------------- #
# Sampler QNN (classical)
# --------------------------------------------------------------------------- #

class SamplerQNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return F.softmax(self.net(inputs), dim=-1)

# --------------------------------------------------------------------------- #
# QCNN (classical)
# --------------------------------------------------------------------------- #

class QCNNModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

# --------------------------------------------------------------------------- #
# Unified hybrid class
# --------------------------------------------------------------------------- #

class GraphQNNGen031:
    """
    Hybrid graph neural network that combines:
    * Classical graph propagation (feedforward)
    * Autoencoder for latent embeddings
    * Sampler network for probabilistic outputs
    * QCNN-style fully connected layers for hierarchical learning

    The class exposes a forward pipeline that accepts a list of input feature tensors,
    propagates them through the classical GNN, encodes latent representations,
    samples distributions, and finally classifies/regresses with QCNN.
    """

    def __init__(self, qnn_arch: Sequence[int], autoencoder_cfg: AutoencoderConfig):
        self.qnn_arch = list(qnn_arch)
        # Build a random classical GNN
        self.weights, self.training_data, self.target_weight = random_network(self.qnn_arch, samples=10)
        # Autoencoder
        self.autoencoder = Autoencoder(
            input_dim=self.qnn_arch[-1],
            latent_dim=autoencoder_cfg.latent_dim,
            hidden_dims=autoencoder_cfg.hidden_dims,
            dropout=autoencoder_cfg.dropout,
        )
        # Sampler QNN
        self.sampler = SamplerQNN()
        # QCNN
        self.qcnn = QCNNModel()

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Execute the full hybrid pipeline.

        Parameters
        ----------
        inputs : List[torch.Tensor]
            List of node feature vectors (each of shape [input_dim]).

        Returns
        -------
        torch.Tensor
            Final QCNN output (probability or regression score).
        """
        # Classical GNN propagation
        activations = feedforward(self.qnn_arch, self.weights, [(x, None) for x in inputs])
        # Take last layer activations as node embeddings
        embeddings = torch.stack([act[-1] for act in activations])
        # Autoencoder latent representation
        latent = self.autoencoder.encode(embeddings)
        # Prepare sampler input: pairwise latent differences
        sampler_inputs = torch.cat([latent[i] - latent[j] for i in range(len(latent)) for j in range(i+1, len(latent))], dim=0)
        sampler_inputs = sampler_inputs.unsqueeze(0)  # batch dim
        probs = self.sampler(sampler_inputs)
        # Use first two probabilities as QCNN input
        qcnn_input = probs[:, :2]
        output = self.qcnn(qcnn_input)
        return output

    def build_graph(self, threshold: float, secondary: float | None = None) -> nx.Graph:
        """
        Construct a graph from the current embeddings using fidelity thresholds.
        """
        embeddings = torch.stack([act[-1] for act in feedforward(self.qnn_arch, self.weights, [(x, None) for x in self.training_data])])
        states = [e.detach() for e in embeddings]
        return fidelity_adjacency(states, threshold, secondary=secondary)

    def train_autoencoder(self, data: torch.Tensor, epochs: int = 50) -> List[float]:
        return train_autoencoder(self.autoencoder, data, epochs=epochs)

    def __repr__(self) -> str:
        return f"<GraphQNNGen031 arch={self.qnn_arch} latent_dim={self.autoencoder.encoder[-1].out_features}>"

__all__ = ["GraphQNNGen031", "AutoencoderConfig", "Autoencoder", "train_autoencoder", "SamplerQNN", "QCNNModel"]
