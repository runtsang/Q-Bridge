import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import networkx as nx
import itertools
from typing import List, Tuple, Sequence, Iterable

class QFCQuantumEncoder(tq.QuantumModule):
    """Quantum encoder that maps 4‑dimensional classical features to a 4‑qubit state."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])

    def forward(self, qdev: tq.QuantumDevice) -> None:
        self.encoder(qdev)

def _random_unitary(n_wires: int, n_ops: int = 30) -> tq.QuantumModule:
    return tq.RandomLayer(n_ops=n_ops, wires=list(range(n_wires)))

def random_network(qnn_arch: Sequence[int], samples: int) -> Tuple[Sequence[int], QFCQuantumEncoder, List[tq.QuantumModule], List[Tuple[torch.Tensor, torch.Tensor]], tq.QuantumModule]:
    encoder = QFCQuantumEncoder()
    layers = [_random_unitary(l_in, l_out) for l_in, l_out in zip(qnn_arch[:-1], qnn_arch[1:])]
    target = _random_unitary(qnn_arch[-1])
    training_data = random_training_data(target, samples)
    return qnn_arch, encoder, layers, training_data, target

def random_training_data(target: tq.QuantumModule, samples: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    data = []
    for _ in range(samples):
        state = torch.randn(4, dtype=torch.complex64).exp()
        state = state / state.norm()
        dev = tq.QuantumDevice(n_wires=4, bsz=1, device="cpu", record_op=True)
        target(dev)
        state_out = dev.get_state()
        data.append((state, state_out))
    return data

def feedforward(qnn_arch: Sequence[int], encoder: QFCQuantumEncoder, layers: Sequence[tq.QuantumModule], samples: Iterable[Tuple[torch.Tensor, torch.Tensor]]) -> List[List[torch.Tensor]]:
    results = []
    for inp, _ in samples:
        dev = tq.QuantumDevice(n_wires=4, bsz=1, device="cpu", record_op=True)
        encoder(dev)
        for layer in layers:
            layer(dev)
        out = dev.get_state()
        results.append([inp, out])
    return results

def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    overlap = torch.vdot(a, b).abs().item()
    return float(overlap ** 2)

def fidelity_adjacency(states: Sequence[torch.Tensor], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(len(states)))
    for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(a, b)
        if fid >= threshold:
            g.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            g.add_edge(i, j, weight=secondary_weight)
    return g

class GraphQNNGen(tq.QuantumModule):
    """Quantum graph neural network that parallels the classical GraphQNNGen."""
    def __init__(self, arch: Sequence[int]):
        super().__init__()
        self.arch = arch
        self.encoder = QFCQuantumEncoder()
        self.layers = nn.ModuleList([ _random_unitary(in_f, out_f) for in_f, out_f in zip(arch[:-1], arch[1:]) ])
        self.norm = nn.BatchNorm1d(arch[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dev = tq.QuantumDevice(n_wires=4, bsz=x.shape[0], device=x.device, record_op=True)
        self.encoder(dev)
        for layer in self.layers:
            layer(dev)
        out = dev.get_state()
        return self.norm(out)

__all__ = [
    "QFCQuantumEncoder",
    "_random_unitary",
    "random_network",
    "random_training_data",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
    "GraphQNNGen",
]
