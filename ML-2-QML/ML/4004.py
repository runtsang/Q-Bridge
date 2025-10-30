"""GraphQNNHybrid – classical implementation.

The class mirrors the quantum interface but operates purely on
NumPy/PyTorch tensors.  It can generate a random feed‑forward
network, run a forward pass, compute state fidelities, build a
fidelity‑based adjacency graph, and train a graph auto‑encoder.
"""
from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union

import networkx as nx
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

Tensor = torch.Tensor


# --------------------------------------------------------------------------- #
# Random network helpers
# --------------------------------------------------------------------------- #
def _rand_lin(in_f: int, out_f: int) -> Tensor:
    return torch.randn(out_f, in_f, dtype=torch.float32)


def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate `samples` (x, y) pairs where y = W @ x."""
    return [(torch.randn(weight.size(1)), weight @ torch.randn(weight.size(1))) for _ in range(samples)]


def random_network(qnn_arch: Sequence[int], samples: int):
    """Return architecture, list of weight tensors, training data and target weight."""
    weights: List[Tensor] = [_rand_lin(in_f, out_f) for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:])]
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight


# --------------------------------------------------------------------------- #
# Forward propagation
# --------------------------------------------------------------------------- #
def feedforward(qnn_arch: Sequence[int],
                weights: Sequence[Tensor],
                samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
    """Return activations for each sample."""
    results: List[List[Tensor]] = []
    for x, _ in samples:
        activations = [x]
        cur = x
        for w in weights:
            cur = torch.tanh(w @ cur)
            activations.append(cur)
        results.append(activations)
    return results


# --------------------------------------------------------------------------- #
# Fidelity utilities
# --------------------------------------------------------------------------- #
def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Squared overlap of two unit‑norm tensors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float((a_norm @ b_norm).item() ** 2)


def fidelity_adjacency(states: Sequence[Tensor],
                       threshold: float,
                       *,
                       secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
    """Build an adjacency graph weighted by state fidelity."""
    G = nx.Graph()
    G.add_nodes_from(range(len(states)))
    for (i, ai), (j, bj) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(ai, bj)
        if fid >= threshold:
            G.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            G.add_edge(i, j, weight=secondary_weight)
    return G


# --------------------------------------------------------------------------- #
# Graph auto‑encoder
# --------------------------------------------------------------------------- #
@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1


class AutoencoderNet(nn.Module):
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        # Encoder
        enc_layers = []
        in_dim = config.input_dim
        for h in config.hidden_dims:
            enc_layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
            if config.dropout > 0:
                enc_layers.append(nn.Dropout(config.dropout))
            in_dim = h
        enc_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder
        dec_layers = []
        in_dim = config.latent_dim
        for h in reversed(config.hidden_dims):
            dec_layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
            if config.dropout > 0:
                dec_layers.append(nn.Dropout(config.dropout))
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def forward(self, x: Tensor) -> Tensor:
        return self.decode(self.encode(x))


def Autoencoder(input_dim: int,
                *,
                latent_dim: int = 32,
                hidden_dims: Tuple[int, int] = (128, 64),
                dropout: float = 0.1) -> AutoencoderNet:
    cfg = AutoencoderConfig(input_dim, latent_dim, hidden_dims, dropout)
    return AutoencoderNet(cfg)


def train_autoencoder(model: AutoencoderNet,
                      data: Tensor,
                      *,
                      epochs: int = 100,
                      batch_size: int = 64,
                      lr: float = 1e-3,
                      weight_decay: float = 0.0,
                      device: torch.device | None = None) -> List[float]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loader = DataLoader(TensorDataset(_as_tensor(data)), batch_size=batch_size, shuffle=True)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for batch, in loader:
            batch = batch.to(device)
            opt.zero_grad(set_to_none=True)
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(loader.dataset)
        history.append(epoch_loss)
    return history


# --------------------------------------------------------------------------- #
# Unified hybrid class
# --------------------------------------------------------------------------- #
class GraphQNNHybrid:
    """Hybrid graph neural network with a classical and a quantum back‑end.

    Public methods are identical for both variants; the constructor flips
    between a PyTorch or a QuTiP implementation based on the `quantum`
    flag.  The class can be used for graph‑based feature extraction,
    fidelity graph construction, or auto‑encoding of graph embeddings.
    """
    def __init__(self, quantum: bool = False):
        self.quantum = quantum

    # ------------------ Classical API  ------------------
    def random_network(self, qnn_arch: Sequence[int], samples: int):
        return random_network(qnn_arch, samples)

    def feedforward(self, qnn_arch: Sequence[int], weights: Sequence[Tensor], samples: Iterable[Tuple[Tensor, Tensor]]):
        return feedforward(qnn_arch, weights, samples)

    def state_fidelity(self, a: Tensor, b: Tensor) -> float:
        return state_fidelity(a, b)

    def fidelity_adjacency(self, states: Sequence[Tensor], threshold: float,
                           *, secondary: float | None = None,
                           secondary_weight: float = 0.5) -> nx.Graph:
        return fidelity_adjacency(states, threshold, secondary=secondary,
                                  secondary_weight=secondary_weight)

    def autoencoder(self, input_dim: int, *, latent_dim: int = 32,
                    hidden_dims: Tuple[int, int] = (128, 64),
                    dropout: float = 0.1) -> AutoencoderNet:
        return Autoencoder(input_dim, latent_dim=latent_dim,
                           hidden_dims=hidden_dims, dropout=dropout)

    def train_autoencoder(self, model: AutoencoderNet, data: Tensor,
                          *, epochs: int = 100, batch_size: int = 64,
                          lr: float = 1e-3, weight_decay: float = 0.0,
                          device: torch.device | None = None) -> List[float]:
        return train_autoencoder(model, data, epochs=epochs, batch_size=batch_size,
                                 lr=lr, weight_decay=weight_decay, device=device)

    # ------------------ Quantum API  ------------------
    def random_network_q(self, qnn_arch: Sequence[int], samples: int):
        """Return architecture, list of unitary lists, training data, target unitary."""
        import qutip as qt
        import scipy as sc
        from math import prod

        def _tensored_id(n):
            I = qt.qeye(2 ** n)
            I.dims = [[2] * n, [2] * n]
            return I

        def _tensored_zero(n):
            P0 = qt.fock(2 ** n).proj()
            P0.dims = [[2] * n, [2] * n]
            return P0

        def _random_unitary(n):
            dim = 2 ** n
            mat = sc.random.normal(size=(dim, dim)) + 1j * sc.random.normal(size=(dim, dim))
            mat, _ = sc.linalg.qr(mat)  # orthonormalize
            q = qt.Qobj(mat)
            q.dims = [[2] * n, [2] * n]
            return q

        target = _random_unitary(qnn_arch[-1])
        train_data = random_training_data_q(target, samples)

        unitaries: List[List[qt.Qobj]] = [[]]
        for layer in range(1, len(qnn_arch)):
            num_in = qnn_arch[layer - 1]
            num_out = qnn_arch[layer]
            layer_ops: List[qt.Qobj] = []
            for out in range(num_out):
                op = _random_unitary(num_in + 1)
                if num_out > 1:
                    op = qt.tensor(_random_unitary(num_in + 1), _tensored_id(num_out - 1))
                    op = self._swap(op, num_in, num_in + out)
                layer_ops.append(op)
            unitaries.append(layer_ops)

        return list(qnn_arch), unitaries, train_data, target

    def feedforward_q(self, qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qt.Qobj]],
                      samples: Iterable[Tuple[qt.Qobj, qt.Qobj]]) -> List[List[qt.Qobj]]:
        """Return state trajectory for each sample."""
        import qutip as qt

        def _layer_state(qnn_arch, unitaries, layer, inp):
            num_in, num_out = qnn_arch[layer - 1], qnn_arch[layer]
            state = qt.tensor(inp, _tensored_zero(num_out))
            U = unitaries[layer][0]
            for g in unitaries[layer][1:]:
                U = g * U
            out_state = U * state * U.dag()
            return self._partial_trace_remove(out_state, list(range(num_in)))

        states: List[List[qt.Qobj]] = []
        for sample, _ in samples:
            traj = [sample]
            cur = sample
            for layer in range(1, len(qnn_arch)):
                cur = _layer_state(qnn_arch, unitaries, layer, cur)
                traj.append(cur)
            states.append(traj)
        return states

    def state_fidelity_q(self, a: qt.Qobj, b: qt.Qobj) -> float:
        return float(abs((a.dag() * b)[0, 0]) ** 2)

    def fidelity_adjacency_q(self, states: Sequence[qt.Qobj], threshold: float,
                             *, secondary: float | None = None,
                             secondary_weight: float = 0.5) -> nx.Graph:
        G = nx.Graph()
        G.add_nodes_from(range(len(states)))
        for (i, ai), (j, bj) in itertools.combinations(enumerate(states), 2):
            fid = self.state_fidelity_q(ai, bj)
            if fid >= threshold:
                G.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                G.add_edge(i, j, weight=secondary_weight)
        return G

    def autoencoder_q(self, num_latent: int, num_trash: int) -> qt.Qobj:
        """Return a QuantumCircuit object mirroring the classical auto‑encoder."""
        from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
        from qiskit.circuit.library import RealAmplitudes
        from qiskit_machine_learning.neural_networks import SamplerQNN
        from qiskit.primitives import StatevectorSampler as Sampler
        from qiskit_machine_learning.optimizers import COBYLA

        circuit = QuantumCircuit(num_latent + 2 * num_trash + 1, 1)
        ansatz = RealAmplitudes(num_latent + num_trash, reps=5)
        circuit.append(ansatz, range(num_latent + num_trash))
        circuit.barrier()
        aux = num_latent + 2 * num_trash
        circuit.h(aux)
        for i in range(num_trash):
            circuit.cswap(aux, num_latent + i, num_latent + num_trash + i)
        circuit.h(aux)
        circuit.measure(aux, 0)
        return circuit  # returned circuit can be wrapped into a SamplerQNN

    # ------------------ Helper utilities ------------------
    @staticmethod
    def _swap(op, src, tgt):
        import qutip as qt
        if src == tgt:
            return op
        order = list(range(len(op.dims[0])))
        order[src], order[tgt] = order[tgt], order[src]
        return op.permute(order)

    @staticmethod
    def _partial_trace_remove(state, remove):
        import qutip as qt
        keep = list(range(len(state.dims[0])))
        for idx in sorted(remove, reverse=True):
            keep.pop(idx)
        return state.ptrace(keep)

    @staticmethod
    def _tensored_zero(num):
        import qutip as qt
        P0 = qt.fock(2 ** num).proj()
        P0.dims = [[2] * num, [2] * num]
        return P0

    @staticmethod
    def _tensored_id(num):
        import qutip as qt
        I = qt.qeye(2 ** num)
        I.dims = [[2] * num, [2] * num]
        return I

    @staticmethod
    def _tensored_zero(num):
        import qutip as qt
        P0 = qt.fock(2 ** num).proj()
        P0.dims = [[2] * num, [2] * num]
        return P0

__all__ = [
    "GraphQNNHybrid",
    "AutoencoderNet",
    "Autoencoder",
    "train_autoencoder",
    "random_network",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
]
