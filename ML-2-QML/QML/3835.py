import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import networkx as nx
import itertools
from typing import List, Tuple, Sequence, Iterable

Tensor = torch.Tensor

def _random_unitary(n_wires: int) -> tq.QuantumModule:
    return tq.RandomLayer(n_ops=20, wires=list(range(n_wires)))

def random_training_data(target_layer: tq.QuantumModule, samples: int):
    data = []
    n_wires = target_layer.n_wires
    for _ in range(samples):
        qdev = tq.QuantumDevice(n_wires=n_wires, bsz=1)
        qdev.initialize()
        rnd = tq.RandomLayer(n_ops=5, wires=list(range(n_wires)))
        rnd(qdev)
        input_state = qdev.state_vector[0].clone()
        target_layer(qdev)
        output_state = qdev.state_vector[0].clone()
        data.append((input_state, output_state))
    return data

def random_quantum_network(arch: Sequence[int], samples: int):
    layers = [_random_unitary(size) for size in arch]
    target_layer = _random_unitary(arch[-1])
    training = random_training_data(target_layer, samples)
    return list(arch), layers, training, target_layer

def feedforward(arch: Sequence[int], layers: Sequence[tq.QuantumModule], samples: Iterable[Tuple[Tensor, Tensor]]):
    states = []
    for input_state,_ in samples:
        qdev = tq.QuantumDevice(n_wires=arch[-1], bsz=1)
        qdev.initialize()
        qdev.state_vector[0] = input_state
        layerwise = [input_state]
        for layer in layers:
            layer(qdev)
            layerwise.append(qdev.state_vector[0].clone())
        states.append(layerwise)
    return states

def state_fidelity(a: Tensor, b: Tensor) -> float:
    return float(abs(torch.vdot(a,b))**2)

def fidelity_adjacency(states: Sequence[Tensor], thresh: float,
                       *, secondary: float | None=None, secondary_w: float=0.5):
    G = nx.Graph()
    G.add_nodes_from(range(len(states)))
    for (i,sa),(j,sb) in itertools.combinations(enumerate(states),2):
        fid = state_fidelity(sa,sb)
        if fid>=thresh:
            G.add_edge(i,j,weight=1.0)
        elif secondary is not None and fid>=secondary:
            G.add_edge(i,j,weight=secondary_w)
    return G

class QLayer(tq.QuantumModule):
    """Simple variational layer combining random and parameterised gates."""
    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        self.random = tq.RandomLayer(n_ops=10, wires=list(range(n_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)
        self.rz = tq.RZ(has_params=True, trainable=True)
        self.cnot = tq.CNOT()

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice):
        self.random(qdev)
        for i in range(self.n_wires):
            self.rx(qdev, wires=i)
        self.ry(qdev, wires=0)
        self.rz(qdev, wires=1)
        self.cnot(qdev, wires=[0,1])

class GraphQNN__gen133(tq.QuantumModule):
    """Hybrid graph‑quantum neural network combining a convolutional encoder and variational layers."""
    def __init__(self, arch: Sequence[int], device='cpu'):
        super().__init__()
        self.arch = arch
        self.n_wires = arch[-1]
        self.encoder = nn.Conv2d(1, self.n_wires, kernel_size=3, padding=1)
        self.q_layers = nn.ModuleList([QLayer(self.n_wires) for _ in arch[1:]])
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    @tq.static_support
    def forward(self, x: torch.Tensor):
        bsz = x.shape[0]
        angles = self.encoder(x).view(bsz, -1)
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        qdev.initialize()
        for i in range(self.n_wires):
            tq.RX()(qdev, wires=i, params=angles[:,i])
        for layer in self.q_layers:
            layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)

    @staticmethod
    def random_instance(arch: Sequence[int], samples: int):
        arch, layers, training, target = random_quantum_network(arch, samples)
        model = GraphQNN__gen133(arch)
        model.q_layers = nn.ModuleList(layers)
        return model, training

__all__ = ['GraphQNN__gen133','feedforward','fidelity_adjacency',
           'random_quantum_network','random_training_data','state_fidelity']
