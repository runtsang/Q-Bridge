"""GraphQNNGen230: unified classical and quantum neural network toolkit.

Classical implementation using PyTorch, NetworkX and NumPy.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable, Sequence, List, Tuple, Callable

import torch
import networkx as nx
import numpy as np
from torch import nn

Tensor = torch.Tensor

# --------------------------------------------------------------------------- #
# Graph‑QNN utilities (classical)
# --------------------------------------------------------------------------- #

def _random_linear(in_features: int, out_features: int) -> Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    dataset = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    weights: List[Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight

def feedforward(qnn_arch: Sequence[int], weights: Sequence[Tensor], samples: Iterable[Tuple[Tensor, Tensor]]):
    activations = []
    for features, _ in samples:
        layer_inputs = features
        layer_outputs = [features]
        for weight in weights:
            layer_inputs = torch.tanh(weight @ layer_inputs)
            layer_outputs.append(layer_inputs)
        activations.append(layer_outputs)
    return activations

def state_fidelity(a: Tensor, b: Tensor) -> float:
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float((a_norm @ b_norm).item() ** 2)

def fidelity_adjacency(states: Sequence[Tensor], threshold: float,
                       *, secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(s_i, s_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

# --------------------------------------------------------------------------- #
# Autoencoder utilities (classical)
# --------------------------------------------------------------------------- #

def _as_tensor(data: Iterable[float] | Tensor) -> Tensor:
    if isinstance(data, Tensor):
        return data
    return torch.as_tensor(data, dtype=torch.float32)

@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

class AutoencoderNet(nn.Module):
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        encoder_layers: List[nn.Module] = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: List[nn.Module] = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, inputs: Tensor) -> Tensor:
        return self.encoder(inputs)

    def decode(self, latents: Tensor) -> Tensor:
        return self.decoder(latents)

    def forward(self, inputs: Tensor) -> Tensor:  # type: ignore[override]
        return self.decode(self.encode(inputs))

def Autoencoder(input_dim: int, *, latent_dim: int = 32,
                hidden_dims: Tuple[int, int] = (128, 64), dropout: float = 0.1) -> AutoencoderNet:
    cfg = AutoencoderConfig(input_dim, latent_dim, hidden_dims, dropout)
    return AutoencoderNet(cfg)

def train_autoencoder(model: AutoencoderNet, data: Tensor, *,
                      epochs: int = 100, batch_size: int = 64,
                      lr: float = 1e-3, weight_decay: float = 0.0,
                      device: torch.device | None = None) -> List[float]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = torch.utils.data.TensorDataset(_as_tensor(data))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: List[float] = []
    for _ in range(epochs):
        epoch_loss = 0.0
        for batch in loader:
            batch = batch[0].to(device)
            optimizer.zero_grad(set_to_none=True)
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

# --------------------------------------------------------------------------- #
# Estimator utilities (classical)
# --------------------------------------------------------------------------- #

class FastBaseEstimator:
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(self, observables: Iterable[Callable[[Tensor], Tensor | float]],
                 parameter_sets: Sequence[Sequence[float]]) -> List[List[float]]:
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, Tensor):
                        val = float(val.mean().cpu())
                    else:
                        val = float(val)
                    row.append(val)
                results.append(row)
        return results

class FastEstimator(FastBaseEstimator):
    def evaluate(self, observables: Iterable[Callable[[Tensor], Tensor | float]],
                 parameter_sets: Sequence[Sequence[float]], *,
                 shots: int | None = None, seed: int | None = None) -> List[List[float]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

# --------------------------------------------------------------------------- #
# QCNN utilities (classical)
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

    def forward(self, inputs: Tensor) -> Tensor:  # type: ignore[override]
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

def QCNN() -> QCNNModel:
    return QCNNModel()

# --------------------------------------------------------------------------- #
# Unified GraphQNNGen230 class
# --------------------------------------------------------------------------- #

class GraphQNNGen230:
    """
    Unified interface that exposes a classical Graph‑QNN, an auto‑encoder, a fast estimator
    and a QCNN‑style network.  The class is intentionally lightweight; all heavy lifting
    is delegated to the sub‑components.
    """

    def __init__(self) -> None:
        self.qnn_arch: List[int] | None = None
        self.weights: List[Tensor] | None = None
        self.training_data: List[Tuple[Tensor, Tensor]] | None = None
        self.target_weight: Tensor | None = None
        self.autoencoder: AutoencoderNet | None = None
        self.estimator: FastBaseEstimator | None = None
        self.qcnn: QCNNModel | None = None

    # ------------------------------------------------------------------ #
    # Graph‑QNN helpers
    # ------------------------------------------------------------------ #

    def build_qnn(self, qnn_arch: Sequence[int], samples: int) -> None:
        self.qnn_arch, self.weights, self.training_data, self.target_weight = random_network(qnn_arch, samples)

    def feedforward(self, samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
        if self.weights is None:
            raise RuntimeError("QNN not built – call build_qnn first.")
        return feedforward(self.qnn_arch, self.weights, samples)

    def fidelity_adjacency(self, threshold: float,
                           *, secondary: float | None = None,
                           secondary_weight: float = 0.5) -> nx.Graph:
        if self.weights is None:
            raise RuntimeError("QNN not built – call build_qnn first.")
        activations = self.feedforward(self.training_data)
        states = [act[-1] for act in activations]
        return fidelity_adjacency(states, threshold, secondary=secondary,
                                  secondary_weight=secondary_weight)

    # ------------------------------------------------------------------ #
    # Autoencoder helpers
    # ------------------------------------------------------------------ #

    def build_autoencoder(self, input_dim: int, *, latent_dim: int = 32,
                          hidden_dims: Tuple[int, int] = (128, 64), dropout: float = 0.1) -> None:
        self.autoencoder = Autoencoder(input_dim, latent_dim=latent_dim,
                                       hidden_dims=hidden_dims, dropout=dropout)

    def train_autoencoder(self, data: Tensor, *, epochs: int = 100,
                          batch_size: int = 64, lr: float = 1e-3,
                          weight_decay: float = 0.0, device: torch.device | None = None) -> List[float]:
        if self.autoencoder is None:
            raise RuntimeError("Autoencoder not built – call build_autoencoder first.")
        return train_autoencoder(self.autoencoder, data, epochs=epochs,
                                 batch_size=batch_size, lr=lr,
                                 weight_decay=weight_decay, device=device)

    # ------------------------------------------------------------------ #
    # Estimator helpers
    # ------------------------------------------------------------------ #

    def build_estimator(self, model: nn.Module) -> None:
        self.estimator = FastEstimator(model)

    def evaluate_estimator(self, observables: Iterable[Callable[[Tensor], Tensor | float]],
                           parameter_sets: Sequence[Sequence[float]],
                           *, shots: int | None = None, seed: int | None = None) -> List[List[float]]:
        if self.estimator is None:
            raise RuntimeError("Estimator not built – call build_estimator first.")
        return self.estimator.evaluate(observables, parameter_sets, shots=shots, seed=seed)

    # ------------------------------------------------------------------ #
    # QCNN helpers
    # ------------------------------------------------------------------ #

    def build_qcnn(self) -> None:
        self.qcnn = QCNN()

    def qcnn_forward(self, inputs: Tensor) -> Tensor:
        if self.qcnn is None:
            raise RuntimeError("QCNN not built – call build_qcnn first.")
        return self.qcnn(inputs)

    __all__ = [
        "GraphQNNGen230",
        "Autoencoder",
        "AutoencoderNet",
        "AutoencoderConfig",
        "train_autoencoder",
        "FastBaseEstimator",
        "FastEstimator",
        "QCNN",
        "QCNNModel",
    ]
