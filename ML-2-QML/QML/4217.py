"""Quantum counterpart of GraphQNNEnhanced using Pennylane."""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import pennylane as qml
import pennylane.numpy as np
import torch

Tensor = torch.Tensor


@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]


class GraphQNNEnhanced:
    """Quantum graph neural network with photonic‑style variational layers
    and a quantum quanvolutional front‑end.  The API mirrors the classical
    counterpart but all computation is performed on a Pennylane device."""

    def __init__(self, qnn_arch: Sequence[int], dev: qml.Device | None = None) -> None:
        self.arch = list(qnn_arch)
        self.dev = dev or qml.device("default.qubit", wires=max(qnn_arch))
        self.params: List[np.ndarray] = []

    # ------------------------------------------------------------------ #
    #  Quantum random network construction
    # ------------------------------------------------------------------ #
    @staticmethod
    def _random_unitary_params(num_wires: int) -> np.ndarray:
        """Return a flat array of rotation angles for a random unitary."""
        return np.random.uniform(-math.pi, math.pi, size=(num_wires,))

    @staticmethod
    def random_network(qnn_arch: Sequence[int], samples: int = 32):
        """Generate architecture, random unitary parameters, training data and target unitary."""
        arch = list(qnn_arch)
        params = [
            GraphQNNEnhanced._random_unitary_params(arch[0]) for _ in range(len(arch) - 1)
        ]
        target = params[-1]
        dataset = [
            (
                np.random.uniform(-math.pi, math.pi, size=arch[0]),
                np.tensordot(target, np.random.uniform(-math.pi, math.pi, size=arch[0]), axes=0),
            )
            for _ in range(samples)
        ]
        return arch, params, dataset, target

    # ------------------------------------------------------------------ #
    #  Quantum feed‑forward
    # ------------------------------------------------------------------ #
    def feedforward(self, samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
        """Run the variational circuit for each sample and record intermediate
        state vectors."""
        results: List[List[Tensor]] = []
        for x, _ in samples:
            circuit = self._build_qnode()
            state = circuit(x)
            results.append([state])
        return results

    def _build_qnode(self):
        @qml.qnode(self.dev, interface="torch")
        def circuit(x):
            # encode input via Ry rotations
            for i in range(x.shape[0]):
                qml.RY(x[i], wires=i)
            # apply photonic‑style layers
            for layer_params in self.params:
                for w, a in zip(range(len(layer_params)), layer_params):
                    qml.RY(a, wires=w)
                # simple entanglement pattern
                for i in range(0, len(layer_params) - 1, 2):
                    qml.CNOT(wires=[i, i + 1])
            return qml.state()
        return circuit

    # ------------------------------------------------------------------ #
    #  Fidelity adjacency using state overlap
    # ------------------------------------------------------------------ #
    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        """Compute squared magnitude of inner product."""
        return float(np.abs(np.vdot(a, b)) ** 2)

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[Tensor], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5
    ) -> nx.Graph:
        G = nx.Graph()
        G.add_nodes_from(range(len(states)))
        for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNNEnhanced.state_fidelity(a, b)
            if fid >= threshold:
                G.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                G.add_edge(i, j, weight=secondary_weight)
        return G

    # ------------------------------------------------------------------ #
    #  Quantum photonic‑style layer construction
    # ------------------------------------------------------------------ #
    @staticmethod
    def _photonic_layer(params: FraudLayerParameters, clip: bool) -> List[float]:
        """Translate photonic parameters into a flat list of rotation angles."""
        def clip_val(v, bound): return max(-bound, min(bound, v))
        angles = [
            params.bs_theta,
            params.bs_phi,
            *params.phases,
            *params.squeeze_r,
            *params.displacement_r,
            *params.kerr,
        ]
        if clip:
            angles = [clip_val(a, 5.0) for a in angles]
        return angles

    def add_fraud_layer(self, params: FraudLayerParameters) -> None:
        """Append a photonic‑style variational layer using mapped angles."""
        self.params.append(self._photonic_layer(params, clip=True))

    # ------------------------------------------------------------------ #
    #  Quantum quanvolution front‑end
    # ------------------------------------------------------------------ #
    class QuantumQuanvolutionFilter:
        """Apply a 2×2 patch quantum kernel to a 28×28 image."""

        def __init__(self, device: qml.Device):
            self.device = device
            self.n_wires = 4
            self.circuit = self._build_circuit()

        def _build_circuit(self) -> qml.QNode:
            @qml.qnode(self.device, interface="torch")
            def circuit(patch: np.ndarray):
                for i, val in enumerate(patch):
                    qml.RY(val, wires=i)
                # simple entanglement pattern
                qml.CNOT(wires=[0, 1])
                qml.CNOT(wires=[2, 3])
                return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1)), \
                       qml.expval(qml.PauliZ(2)), qml.expval(qml.PauliZ(3))
            return circuit

        def __call__(self, image: np.ndarray) -> np.ndarray:
            h, w = image.shape
            patches = []
            for r in range(0, h, 2):
                for c in range(0, w, 2):
                    patch = image[r:r + 2, c:c + 2].flatten()
                    out = self.circuit(patch)
                    patches.append(out)
            return np.concatenate(patches)

    def quanvolution_forward(self, image: np.ndarray) -> np.ndarray:
        """Apply the quantum quanvolution filter to a single image."""
        filter = self.QuantumQuanvolutionFilter(self.dev)
        return filter(image)

    # ------------------------------------------------------------------ #
    #  Convenience helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def random_training_data(unitary: np.ndarray, samples: int) -> List[Tuple[Tensor, Tensor]]:
        dataset: List[Tuple[Tensor, Tensor]] = []
        for _ in range(samples):
            state = np.random.randn(unitary.shape[0]) + 1j * np.random.randn(unitary.shape[0])
            state = state / np.linalg.norm(state)
            target = unitary @ state
            dataset.append((state, target))
        return dataset


__all__ = ["GraphQNNEnhanced", "FraudLayerParameters"]
