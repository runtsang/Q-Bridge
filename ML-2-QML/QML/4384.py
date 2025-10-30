import numpy as np
import torch
import torchquantum as tq
import torchquantum.functional as tqf
import strawberryfields as sf
from strawberryfields import Engine
from strawberryfields.ops import Dgate, BSgate, Rgate, Sgate, Kgate
import networkx as nx
import itertools
from typing import Sequence, Iterable, Tuple
from dataclasses import dataclass

# --- FraudDetection utilities -------------------------------------------------
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

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> sf.Program:
    """Create a Strawberry Fields program for the hybrid fraud detection model."""
    program = sf.Program(2)
    with program.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
    return program

def _apply_layer(modes: Sequence, params: FraudLayerParameters, *, clip: bool) -> None:
    BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
    for i, phase in enumerate(params.phases):
        Rgate(phase) | modes[i]
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        Sgate(r if not clip else _clip(r, 5), phi) | modes[i]
    BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
    for i, phase in enumerate(params.phases):
        Rgate(phase) | modes[i]
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        Dgate(r if not clip else _clip(r, 5), phi) | modes[i]
    for i, k in enumerate(params.kerr):
        Kgate(k if not clip else _clip(k, 1)) | modes[i]

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

# --- GraphQNN utilities -------------------------------------------------------
def _random_linear(in_features: int, out_features: int) -> torch.Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: torch.Tensor, samples: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    dataset: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    weights: list[torch.Tensor] = []
    for in_features, out_features in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_features, out_features))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight

def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[torch.Tensor],
    samples: Iterable[tuple[torch.Tensor, torch.Tensor]],
) -> list[list[torch.Tensor]]:
    stored: list[list[torch.Tensor]] = []
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

# --- QuantumKernelMethodGen ----------------------------------------------------
class QuantumKernelMethodGen(tq.QuantumModule):
    """Quantum kernel method with multiple ansatz options.

    Supports a programmable RY ansatz for qubit encoding and a photonic
    circuit built with Strawberry Fields.  A convolutional encoder
    (inspired by QFCModel) maps images to a 4‑dimensional vector before
    kernel evaluation.  Graph‑based weighting via state fidelities is
    optionally applied.
    """

    def __init__(self, use_photonic: bool = False,
                 graph_threshold: float = 0.8, gamma: float = 1.0) -> None:
        super().__init__()
        self.use_photonic = use_photonic
        self.graph_threshold = graph_threshold
        self.gamma = gamma
        if use_photonic:
            self.n_wires = 2
            self.q_device = sf.Program(self.n_wires)
        else:
            self.n_wires = 4
            self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
            self.ansatz = self._build_qubit_ansatz()
        # Convolutional encoder
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.measure = tq.MeasureAll(tq.PauliZ)

    def _build_qubit_ansatz(self) -> tq.QuantumModule:
        class Ans(tq.QuantumModule):
            def __init__(self):
                super().__init__()
                self.n_wires = 4
                self.random_layer = tq.RandomLayer(n_ops=20, wires=list(range(4)))
                self.rx = tq.RX(has_params=True, trainable=True)
                self.ry = tq.RY(has_params=True, trainable=True)
                self.rz = tq.RZ(has_params=True, trainable=True)

            @tq.static_support
            def forward(self, qdev: tq.QuantumDevice):
                self.random_layer(qdev)
                self.rx(qdev, wires=0)
                self.ry(qdev, wires=1)
                self.rz(qdev, wires=2)
                tqf.hadamard(qdev, wires=3, static=True)
        return Ans()

    def _photonic_ansatz(self, qprog: sf.Program, data: torch.Tensor) -> None:
        with qprog.context as q:
            for i, amp in enumerate(data):
                Dgate(amp.item(), 0) | q[i]
            BSgate(np.pi/4, 0) | (q[0], q[1])

    def _graph_adjacency(self, states: Sequence[torch.Tensor]) -> nx.Graph:
        return fidelity_adjacency(states, self.graph_threshold)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        if self.use_photonic:
            # Photonic overlap via Strawberry Fields
            prog = sf.Program(self.n_wires)
            self._photonic_ansatz(prog, x[0])
            self._photonic_ansatz(prog, y[0])
            eng = Engine("fock", backend_options={"cutoff_dim": 5})
            result = eng.run(prog)
            state = result.state
            overlap = state.overlap(state)
            return torch.tensor(overlap, dtype=torch.float32)
        else:
            self.q_device.reset_states(bsz)
            self.ansatz(self.q_device)
            # Encode x and y with opposite signs
            self.encoder(self.q_device, x)
            self.encoder(self.q_device, -y)
            out = torch.abs(self.q_device.states.view(-1)[0])
            return out

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        mat = np.zeros((len(a), len(b)))
        for i, x in enumerate(a):
            for j, y in enumerate(b):
                mat[i, j] = self.forward(x.unsqueeze(0), y.unsqueeze(0)).item()
        return mat

    # Expose GraphQNN utilities
    @staticmethod
    def random_network(qnn_arch: Sequence[int], samples: int):
        return random_network(qnn_arch, samples)

    @staticmethod
    def feedforward(qnn_arch: Sequence[int], weights: Sequence[torch.Tensor],
                    samples: Iterable[tuple[torch.Tensor, torch.Tensor]]):
        return feedforward(qnn_arch, weights, samples)

    @staticmethod
    def fidelity_adjacency(states: Sequence[torch.Tensor], threshold: float,
                           *, secondary: float | None = None,
                           secondary_weight: float = 0.5) -> nx.Graph:
        return fidelity_adjacency(states, threshold, secondary=secondary,
                                   secondary_weight=secondary_weight)

    @staticmethod
    def build_fraud_detection_program(
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
    ) -> sf.Program:
        return build_fraud_detection_program(input_params, layers)
