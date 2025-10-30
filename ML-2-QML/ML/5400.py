from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Iterable, List, Sequence, Tuple, Callable
import numpy as np
import networkx as nx

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class HybridSamplerQNN(nn.Module):
    """
    Hybrid classical‑quantum sampler that combines:
    * an MLP auto‑encoder (encoder/decoder) for dimensionality reduction
    * a quantum sampler circuit (via qiskit) for generating probability distributions
    * a FastBaseEstimator‑style evaluator for batch inference
    * a graph‑based adjacency built on state fidelities
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: Tuple[int, int] = (128, 64),
        dropout: float = 0.1,
        quantum_sampler: Any | None = None,
    ) -> None:
        super().__init__()
        # Auto‑encoder backbone
        encoder_layers: List[nn.Module] = []
        in_dim = input_dim
        for h in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, h))
            encoder_layers.append(nn.ReLU())
            if dropout > 0.0:
                encoder_layers.append(nn.Dropout(dropout))
            in_dim = h
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: List[nn.Module] = []
        in_dim = latent_dim
        for h in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, h))
            decoder_layers.append(nn.ReLU())
            if dropout > 0.0:
                decoder_layers.append(nn.Dropout(dropout))
            in_dim = h
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        self.quantum_sampler = quantum_sampler  # placeholder for a qiskit SamplerQNN

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # classical encoding/decoding
        z = self.encode(x)
        recon = self.decode(z)
        # if a quantum sampler is attached, fuse its output
        if self.quantum_sampler is not None:
            # assume sampler expects 2‑dimensional latent vector
            latent_np = z.detach().cpu().numpy()
            samp = self.quantum_sampler.sample(latent_np)
            samp = torch.as_tensor(samp, dtype=torch.float32)
            return samp
        return recon

    # FastBaseEstimator style evaluation
    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []
        self.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self(inputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        val = float(val.mean().cpu())
                    else:
                        val = float(val)
                    row.append(val)
                results.append(row)
        return results

    # Graph adjacency based on latent state fidelities
    def latent_fidelity_graph(
        self,
        inputs: torch.Tensor,
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        z = self.encode(inputs)
        states = [self._vector_to_state(v) for v in z]
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for i, a in enumerate(states):
            for j, b in enumerate(states[i + 1 :], i + 1):
                fid = self._state_fidelity(a, b)
                if fid >= threshold:
                    graph.add_edge(i, j, weight=1.0)
                elif secondary is not None and fid >= secondary:
                    graph.add_edge(i, j, weight=secondary_weight)
        return graph

    @staticmethod
    def _vector_to_state(v: torch.Tensor) -> torch.Tensor:
        # normalize
        return v / (torch.norm(v) + 1e-12)

    @staticmethod
    def _state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
        a_n = a / (torch.norm(a) + 1e-12)
        b_n = b / (torch.norm(b) + 1e-12)
        return float(torch.dot(a_n, b_n).item() ** 2)


def SamplerQNN(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    quantum_sampler: Any | None = None,
) -> HybridSamplerQNN:
    """Factory mirroring the quantum helper returning a configured hybrid network."""
    return HybridSamplerQNN(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        quantum_sampler=quantum_sampler,
    )


__all__ = ["HybridSamplerQNN", "SamplerQNN"]
