from __future__ import annotations
from typing import Sequence, Iterable, Tuple, List
import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
import networkx as nx
import itertools
import scipy as sc

# ----------------- quantum kernel utilities ----------------- #
class _QuantumEncoder(tq.QuantumModule):
    """Encode classical data into a quantum state via Ry rotations."""
    def __init__(self, n_wires: int) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        self.encoder(q_device, x)

class _QuantumKernel(tq.QuantumModule):
    """Quantum kernel based on overlap of two encoded states."""
    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.encoder = _QuantumEncoder(self.n_wires)
        self.random_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    @tq.static_support
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """
        Compute the kernel value for a single pair of samples.
        The circuit is: encode x -> random_layer -> encode -y -> random_layer.
        """
        batch = x.shape[0]
        self.q_device.reset_states(batch)
        # encode x
        self.encoder(self.q_device, x)
        self.random_layer(self.q_device)
        # encode -y
        self.encoder(self.q_device, -y)
        self.random_layer(self.q_device)
        # measurement
        self.measure(self.q_device)

    def kernel_value(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return a scalar kernel value for two batch‑wise tensors."""
        self.forward(x, y)
        # the overlap is the absolute square of the first amplitude
        return torch.abs(self.q_device.states.view(-1)[0]).unsqueeze(0)

def kernel_matrix(a: Sequence[torch.Tensor],
                  b: Sequence[torch.Tensor],
                  n_wires: int = 4) -> np.ndarray:
    kernel = _QuantumKernel(n_wires)
    return np.array([[kernel.kernel_value(x, y).item() for y in b] for x in a])

# ----------------- quantum fully‑connected layer ----------------- #
class _QuantumFCL(tq.QuantumModule):
    """Parameterised one‑qubit circuit that mimics a fully‑connected layer."""
    def __init__(self, n_qubits: int = 1) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.q_device = tq.QuantumDevice(n_wires=self.n_qubits)

    @tq.static_support
    def forward(self, thetas: Iterable[float]) -> torch.Tensor:
        thetas = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        self.q_device.reset_states(thetas.shape[0])
        tq.ry(self.q_device, wires=range(self.n_qubits), params=thetas)
        return torch.tanh(self.q_device.states.view(-1))

# ----------------- quantum quanvolution filter ----------------- #
class _QuantumQuanvolution(tq.QuantumModule):
    """Apply a random two‑qubit circuit to all 2×2 patches of a 28×28 image."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.encoder = _QuantumEncoder(self.n_wires)
        self.random_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    @tq.static_support
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        x = x.view(bsz, 28, 28)
        patches: List[torch.Tensor] = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, patch)
                self.random_layer(qdev)
                meas = self.measure(qdev)
                patches.append(meas.view(bsz, 4))
        return torch.cat(patches, dim=1)

# ----------------- hybrid quantum‑classical integration ----------------- #
class QuantumKernelMethod(tq.QuantumModule):
    """Hybrid quantum kernel module that combines a quantum kernel,
    a quantum fully‑connected layer, and a quantum quanvolution filter.
    It can be used as a drop‑in replacement for the classical
    :class:`QuantumKernelMethod` in hybrid training pipelines.
    """
    def __init__(self,
                 n_wires: int = 4,
                 head_features: int = 10) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.kernel = _QuantumKernel(n_wires)
        self.fcl = _QuantumFCL(n_qubits=1)
        self.quanvolution = _QuantumQuanvolution()
        # The output of the quanvolution has 4×14×14 = 784 features
        self.linear_head = torch.nn.Linear(4 * 14 * 14, head_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a batch through the quanvolution filter and linear head."""
        features = self.quanvolution(x)
        logits = self.linear_head(features)
        return torch.nn.functional.log_softmax(logits, dim=-1)

    def kernel_matrix(self,
                      a: Sequence[torch.Tensor],
                      b: Sequence[torch.Tensor]) -> np.ndarray:
        return kernel_matrix(a, b, n_wires=self.n_wires)

    def fidelity_adjacency(self,
                           states: Sequence[torch.Tensor],
                           threshold: float,
                           *,
                           secondary: float | None = None,
                           secondary_weight: float = 0.5) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, ai), (j, aj) in itertools.combinations(enumerate(states), 2):
            fid = torch.abs((ai / (torch.norm(ai) + 1e-12)).dot(aj / (torch.norm(aj) + 1e-12)))**2
            fid_val = fid.item()
            if fid_val >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid_val >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    def random_quantum_network(self, arch: Sequence[int], samples: int):
        """Generate a toy quantum network with random unitaries and training data."""
        def random_unitary(dim: int) -> torch.Tensor:
            mat = torch.randn(dim, dim, dtype=torch.complex64)
            q, r = torch.linalg.qr(mat)
            diag = torch.diagonal(r)
            phase = diag / torch.abs(diag)
            q = q * phase
            return q

        unitaries: List[List[torch.Tensor]] = [[]]
        for layer in range(1, len(arch)):
            for _ in range(arch[layer]):
                unitaries.append([random_unitary(arch[layer-1] + 1)])
        target = random_unitary(arch[-1])
        data = []
        for _ in range(samples):
            state = torch.randn(arch[0], dtype=torch.complex64)
            state = state / torch.norm(state)
            transformed = target @ state
            data.append((state, transformed))
        return list(arch), unitaries, data, target

    def random_training_data(self, samples: int):
        _, _, data, _ = self.random_quantum_network([1], samples)
        return data

# Preserve backward‑compatible names from the original module
KernalAnsatz = _QuantumEncoder
Kernel = _QuantumKernel
__all__ = [
    "KernalAnsatz",
    "Kernel",
    "kernel_matrix",
    "QuantumKernelMethod",
]
