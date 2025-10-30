from __future__ import annotations

import itertools
from typing import Iterable, Sequence, Tuple

import networkx as nx
import qutip as qt
import scipy as sc
import torch
import torch.nn as nn
import torch.quantum as tq  # placeholder for torchquantum
import torch.quantum.functional as tqf  # placeholder
from qiskit.circuit import Parameter
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning import algorithm_globals
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorSampler as Sampler

Tensor = torch.Tensor

# --- Quantum utilities (from GraphQNN.py) --------------------------------
def _tensored_id(num_qubits: int) -> qt.Qobj:
    identity = qt.qeye(2 ** num_qubits)
    dims = [2] * num_qubits
    identity.dims = [dims.copy(), dims.copy()]
    return identity

def _tensored_zero(num_qubits: int) -> qt.Qobj:
    projector = qt.fock(2 ** num_qubits).proj()
    dims = [2] * num_qubits
    projector.dims = [dims.copy(), dims.copy()]
    return projector

def _swap_registers(op: qt.Qobj, source: int, target: int) -> qt.Qobj:
    if source == target:
        return op
    order = list(range(len(op.dims[0])))
    order[source], order[target] = order[target], order[source]
    return op.permute(order)

def _random_qubit_unitary(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    matrix = sc.random.normal(size=(dim, dim)) + 1j * sc.random.normal(size=(dim, dim))
    unitary = sc.linalg.orth(matrix)
    qobj = qt.Qobj(unitary)
    dims = [2] * num_qubits
    qobj.dims = [dims.copy(), dims.copy()]
    return qobj

def _random_qubit_state(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    amplitudes = sc.random.normal(size=(dim, 1)) + 1j * sc.random.normal(size=(dim, 1))
    amplitudes /= sc.linalg.norm(amplitudes)
    state = qt.Qobj(amplitudes)
    state.dims = [[2] * num_qubits, [1] * num_qubits]
    return state

def random_training_data(unitary: qt.Qobj, samples: int) -> list[tuple[qt.Qobj, qt.Qobj]]:
    dataset = []
    num_qubits = len(unitary.dims[0])
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        dataset.append((state, unitary * state))
    return dataset

def random_network(qnn_arch: list[int], samples: int):
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    unitaries: list[list[qt.Qobj]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        layer_ops: list[qt.Qobj] = []
        for output in range(num_outputs):
            op = _random_qubit_unitary(num_inputs + 1)
            if num_outputs > 1:
                op = qt.tensor(_random_qubit_unitary(num_inputs + 1), _tensored_id(num_outputs - 1))
                op = _swap_registers(op, num_inputs, num_inputs + output)
            layer_ops.append(op)
        unitaries.append(layer_ops)

    return qnn_arch, unitaries, training_data, target_unitary

def _partial_trace_keep(state: qt.Qobj, keep: Sequence[int]) -> qt.Qobj:
    if len(keep)!= len(state.dims[0]):
        return state.ptrace(list(keep))
    return state

def _partial_trace_remove(state: qt.Qobj, remove: Sequence[int]) -> qt.Qobj:
    keep = list(range(len(state.dims[0])))
    for index in sorted(remove, reverse=True):
        keep.pop(index)
    return _partial_trace_keep(state, keep)

def _layer_channel(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qt.Qobj]], layer: int, input_state: qt.Qobj) -> qt.Qobj:
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    state = qt.tensor(input_state, _tensored_zero(num_outputs))

    layer_unitary = unitaries[layer][0].copy()
    for gate in unitaries[layer][1:]:
        layer_unitary = gate * layer_unitary

    return _partial_trace_remove(layer_unitary * state * layer_unitary.dag(), range(num_inputs))

def feedforward(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qt.Qobj]], samples: Iterable[tuple[qt.Qobj, qt.Qobj]]):
    stored_states = []
    for sample, _ in samples:
        layerwise = [sample]
        current_state = sample
        for layer in range(1, len(qnn_arch)):
            current_state = _layer_channel(qnn_arch, unitaries, layer, current_state)
            layerwise.append(current_state)
        stored_states.append(layerwise)
    return stored_states

def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
    return abs((a.dag() * b)[0, 0]) ** 2

def fidelity_adjacency(states: Sequence[qt.Qobj], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

# --- Quantum EstimatorQNN -------------------------------------------------------
def EstimatorQNN() -> QiskitEstimatorQNN:
    params1 = [Parameter("input1"), Parameter("weight1")]
    qc1 = QuantumCircuit(1)
    qc1.h(0)
    qc1.ry(params1[0], 0)
    qc1.rx(params1[1], 0)

    observable1 = SparsePauliOp.from_list([("Y" * qc1.num_qubits, 1)])

    estimator = Estimator()
    estimator_qnn = QiskitEstimatorQNN(
        circuit=qc1,
        observables=observable1,
        input_params=[params1[0]],
        weight_params=[params1[1]],
        estimator=estimator,
    )
    return estimator_qnn

# --- Quantum Autoencoder (SamplerQNN) --------------------------------------------
def Autoencoder() -> SamplerQNN:
    algorithm_globals.random_seed = 42
    sampler = Sampler()

    def ansatz(num_qubits):
        return RealAmplitudes(num_qubits, reps=5)

    def auto_encoder_circuit(num_latent, num_trash):
        qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        circuit = QuantumCircuit(qr, cr)
        circuit.compose(ansatz(num_latent + num_trash), range(0, num_latent + num_trash), inplace=True)
        circuit.barrier()
        auxiliary_qubit = num_latent + 2 * num_trash
        circuit.h(auxiliary_qubit)
        for i in range(num_trash):
            circuit.cswap(auxiliary_qubit, num_latent + i, num_latent + num_trash + i)
        circuit.h(auxiliary_qubit)
        circuit.measure(auxiliary_qubit, cr[0])
        return circuit

    num_latent = 3
    num_trash = 2
    qc = auto_encoder_circuit(num_latent, num_trash)

    def identity_interpret(x):
        return x

    qnn = SamplerQNN(
        circuit=qc,
        input_params=[],
        weight_params=qc.parameters,
        interpret=identity_interpret,
        output_shape=2,
        sampler=sampler,
    )
    return qnn

# --- Quantum LSTM (torchquantum) ------------------------------------------------
class QLSTM(nn.Module):
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [0], "func": "rx", "wires": [0]},
                    {"input_idx": [1], "func": "rx", "wires": [1]},
                    {"input_idx": [2], "func": "rx", "wires": [2]},
                    {"input_idx": [3], "func": "rx", "wires": [3]},
                ]
            )
            self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            for wire in range(self.n_wires):
                if wire == self.n_wires - 1:
                    tqf.cnot(qdev, wires=[wire, 0])
                else:
                    tqf.cnot(qdev, wires=[wire, wire + 1])
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        self.forget = self.QLayer(n_qubits)
        self.input = self.QLayer(n_qubits)
        self.update = self.QLayer(n_qubits)
        self.output = self.QLayer(n_qubits)

        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(self.linear_forget(combined)))
            i = torch.sigmoid(self.input(self.linear_input(combined)))
            g = torch.tanh(self.update(self.linear_update(combined)))
            o = torch.sigmoid(self.output(self.linear_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

# --- Hybrid GraphQNNGen ------------------------------------------------------------
class GraphQNNGen:
    """Quantum‑enhanced hybrid graph neural network combining
    a quantum graph network, a quantum estimator, a quantum autoencoder,
    and a quantum LSTM for sequential graph data.
    """

    def __init__(
        self,
        graph_arch: Sequence[int],
        estimator_arch: Sequence[int],
        lstm_cfg: Tuple[int, int],  # (input_dim, hidden_dim)
    ) -> None:
        # Quantum graph network
        self.qnn_arch = list(graph_arch)
        self.qnn_weights = [_random_qubit_unitary(in_f) for in_f in graph_arch[:-1]]

        # Estimator
        self.estimator = EstimatorQNN()

        # Autoencoder
        self.autoencoder = Autoencoder()

        # LSTM
        self.lstm = QLSTM(lstm_cfg[0], lstm_cfg[1], n_qubits=4)

    def forward(self, graph: nx.Graph, node_states: Sequence[qt.Qobj]) -> qt.Qobj:
        # 1. Fidelity adjacency from quantum states
        adj = fidelity_adjacency(node_states, threshold=0.8)

        # 2. Quantum feed‑forward on graph network
        samples = [(state, qt.Qobj(0)) for state in node_states]
        ff_outputs = feedforward(self.qnn_arch, self.qnn_weights, samples)

        # 3. Autoencoder latent representation (quantum)
        latent_states = [self.autoencoder([state]) for state in node_states]

        # 4. Sequence modeling via quantum LSTM on latent states
        latent_tensors = torch.tensor([s.full().flatten() for s in latent_states], dtype=torch.complex64)
        lstm_out, _ = self.lstm(latent_tensors.unsqueeze(0))

        return lstm_out

__all__ = [
    "GraphQNNGen",
    "EstimatorQNN",
    "Autoencoder",
    "QLSTM",
    "fidelity_adjacency",
    "feedforward",
]
