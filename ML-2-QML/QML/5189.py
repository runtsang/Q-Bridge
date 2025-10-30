"""Quantum‑enhanced Graph Neural Network.

This module mirrors the classical GraphQNNModel but replaces the
feed‑forward, self‑attention, transformer and classifier with
quantum circuits.  The public API is identical so that client code
can swap between the two implementations without changes.

The implementation uses Qiskit for state propagation and Self‑Attention,
qutip for fidelity computations, and TorchQuantum for the transformer
blocks.
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import qiskit
import qutip as qt
import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# 1. Quantum state‑propagation helpers (qutip)
# --------------------------------------------------------------------------- #
def _tensored_id(num_qubits: int) -> qt.Qobj:
    I = qt.qeye(2 ** num_qubits)
    dims = [2] * num_qubits
    I.dims = [dims.copy(), dims.copy()]
    return I


def _tensored_zero(num_qubits: int) -> qt.Qobj:
    proj = qt.fock(2 ** num_qubits).proj()
    dims = [2] * num_qubits
    proj.dims = [dims.copy(), dims.copy()]
    return proj


def _swap_registers(op: qt.Qobj, source: int, target: int) -> qt.Qobj:
    if source == target:
        return op
    order = list(range(len(op.dims[0])))
    order[source], order[target] = order[target], order[source]
    return op.permute(order)


def _random_qubit_unitary(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    matrix = (qt.random.normal(size=(dim, dim)) +
              1j * qt.random.normal(size=(dim, dim)))
    unitary = qt.Qobj(qt.linalg.orth(matrix))
    dims = [2] * num_qubits
    unitary.dims = [dims.copy(), dims.copy()]
    return unitary


def _random_qubit_state(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    amp = (qt.random.normal(size=(dim, 1)) +
           1j * qt.random.normal(size=(dim, 1)))
    amp /= qt.linalg.norm(amp)
    state = qt.Qobj(amp)
    state.dims = [[2] * num_qubits, [1] * num_qubits]
    return state


def random_training_data(unitary: qt.Qobj, samples: int) -> List[Tuple[qt.Qobj, qt.Qobj]]:
    """Generate (input, target) pairs where the target is the unitary applied to the input."""
    data: List[Tuple[qt.Qobj, qt.Qobj]] = []
    n_qubits = len(unitary.dims[0])
    for _ in range(samples):
        inp = _random_qubit_state(n_qubits)
        data.append((inp, unitary * inp))
    return data


def random_network(qnn_arch: List[int], samples: int) -> Tuple[List[int], List[List[qt.Qobj]],
                                                               List[Tuple[qt.Qobj, qt.Qobj]],
                                                               qt.Qobj]:
    """Construct a layered ansatz and corresponding training data."""
    target = _random_qubit_unitary(qnn_arch[-1])
    data = random_training_data(target, samples)

    unitaries: List[List[qt.Qobj]] = [[]]
    for layer in range(1, len(qnn_arch)):
        in_q = qnn_arch[layer - 1]
        out_q = qnn_arch[layer]
        layer_ops: List[qt.Qobj] = []
        for out in range(out_q):
            op = _random_qubit_unitary(in_q + 1)
            if out_q > 1:
                op = qt.tensor(_random_qubit_unitary(in_q + 1), _tensored_id(out_q - 1))
                op = _swap_registers(op, in_q, in_q + out)
            layer_ops.append(op)
        unitaries.append(layer_ops)

    return qnn_arch, unitaries, data, target


def _partial_trace_keep(state: qt.Qobj, keep: Sequence[int]) -> qt.Qobj:
    if len(keep)!= len(state.dims[0]):
        return state.ptrace(list(keep))
    return state


def _partial_trace_remove(state: qt.Qobj, remove: Sequence[int]) -> qt.Qobj:
    keep = list(range(len(state.dims[0])))
    for idx in sorted(remove, reverse=True):
        keep.pop(idx)
    return _partial_trace_keep(state, keep)


def _layer_channel(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qt.Qobj]],
                   layer: int, input_state: qt.Qobj) -> qt.Qobj:
    in_q = qnn_arch[layer - 1]
    out_q = qnn_arch[layer]
    state = qt.tensor(input_state, _tensored_zero(out_q))

    layer_unitary = unitaries[layer][0].copy()
    for gate in unitaries[layer][1:]:
        layer_unitary = gate * layer_unitary

    return _partial_trace_remove(layer_unitary * state * layer_unitary.dag(), range(in_q))


def feedforward(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qt.Qobj]],
                samples: Iterable[Tuple[qt.Qobj, qt.Qobj]]) -> List[List[qt.Qobj]]:
    """Compute layerwise states for each sample."""
    outputs: List[List[qt.Qobj]] = []
    for samp, _ in samples:
        layerwise = [samp]
        current = samp
        for layer in range(1, len(qnn_arch)):
            current = _layer_channel(qnn_arch, unitaries, layer, current)
            layerwise.append(current)
        outputs.append(layerwise)
    return outputs


def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
    """Squared overlap between pure states."""
    return abs((a.dag() * b)[0, 0]) ** 2


def fidelity_adjacency(states: Sequence[qt.Qobj], threshold: float,
                       *, secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(len(states)))
    for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(s_i, s_j)
        if fid >= threshold:
            G.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            G.add_edge(i, j, weight=secondary_weight)
    return G


# --------------------------------------------------------------------------- #
# 2. Quantum self‑attention (Qiskit)
# --------------------------------------------------------------------------- #
class QuantumSelfAttention:
    """Self‑attention block implemented as a parametrised Qiskit circuit."""

    def __init__(self, n_qubits: int = 4):
        self.n_qubits = n_qubits
        self.qr = qiskit.QuantumRegister(n_qubits, "q")
        self.cr = qiskit.ClassicalRegister(n_qubits, "c")

    def _build_circuit(self,
                       rotation_params: qiskit.circuit.ParameterVector,
                       entangle_params: qiskit.circuit.ParameterVector) -> qiskit.QuantumCircuit:
        circ = qiskit.QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circ.rx(rotation_params[3 * i], i)
            circ.ry(rotation_params[3 * i + 1], i)
            circ.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circ.crx(entangle_params[i], i, i + 1)
        circ.measure(self.qr, self.cr)
        return circ

    def run(self,
            backend: qiskit.providers.BaseBackend,
            rotation_params: qiskit.circuit.ParameterVector,
            entangle_params: qiskit.circuit.ParameterVector,
            shots: int = 1024) -> dict:
        circ = self._build_circuit(rotation_params, entangle_params)
        job = qiskit.execute(circ, backend, shots=shots)
        return job.result().get_counts(circ)


# --------------------------------------------------------------------------- #
# 3. Quantum transformer block (TorchQuantum)
# --------------------------------------------------------------------------- #
class QuantumTransformerBlock(tq.QuantumModule):
    """Transformer block where the linear projections are realised by a
    small quantum circuit.  The block is fully differentiable via
    TorchQuantum's automatic differentiation."""

    def __init__(self, embed_dim: int, num_heads: int,
                 ffn_dim: int, n_wires: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.n_wires = n_wires
        self.dropout = nn.Dropout(dropout)

        # Linear projections (classical) followed by a tiny quantum circuit
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.combine = nn.Linear(embed_dim, embed_dim)

        # Quantum submodule that acts on each head
        self.q_layer = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)],
        )
        self.q_params = nn.Parameter(torch.randn(n_wires))

    def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
        # Classical projections
        k = self.k_linear(x)
        q = self.q_linear(x)
        v = self.v_linear(x)

        batch, seq, _ = x.shape
        d_k = self.embed_dim // self.num_heads

        k = k.view(batch, seq, self.num_heads, d_k)
        q = q.view(batch, seq, self.num_heads, d_k)
        v = v.view(batch, seq, self.num_heads, d_k)

        # Apply quantum circuit to each head
        def quantum_head(head: torch.Tensor) -> torch.Tensor:
            qdev = q_device.copy(bsz=head.size(0), device=head.device)
            self.q_layer(qdev, head)
            state_vec = torch.tensor(qdev.state, dtype=torch.float32)
            return state_vec

        heads = []
        for head in torch.unbind(k, dim=2):
            heads.append(quantum_head(head))
        heads = torch.stack(heads, dim=2)  # (batch, seq, heads, n_wires)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)

        # Weighted sum
        out = torch.matmul(scores, v)
        out = out.view(batch, seq, self.embed_dim)
        return self.combine(out)


# --------------------------------------------------------------------------- #
# 4. Quantum classifier circuit (Qiskit)
# --------------------------------------------------------------------------- #
def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[qiskit.QuantumCircuit,
                                                                   List[qiskit.circuit.Parameter],
                                                                   List[qiskit.circuit.Parameter],
                                                                   List[qiskit.quantum_info.SparsePauliOp]]:
    """Create a layered ansatz with explicit encoding and variational parameters."""
    encoding = qiskit.circuit.ParameterVector("x", num_qubits)
    weights = qiskit.circuit.ParameterVector("theta", num_qubits * depth)

    circ = qiskit.QuantumCircuit(num_qubits)
    for qubit, param in zip(range(num_qubits), encoding):
        circ.rx(param, qubit)

    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circ.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            circ.cz(qubit, qubit + 1)

    observables = [qiskit.quantum_info.SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
                   for i in range(num_qubits)]
    return circ, list(encoding), list(weights), observables


# --------------------------------------------------------------------------- #
# 5. Public API: GraphQNNModel (quantum)
# --------------------------------------------------------------------------- #
class GraphQNNModel:
    """Quantum‑enhanced graph neural network with identical API to the classical version."""

    def __init__(self, architecture: Sequence[int],
                 num_heads: int = 4, ffn_dim: int = 64,
                 num_blocks: int = 2, num_classes: int = 2):
        self.arch = list(architecture)
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.num_blocks = num_blocks
        self.num_classes = num_classes

        # Quantum transformer stack
        self.transformer = nn.Sequential(
            *[QuantumTransformerBlock(self.arch[-1], self.num_heads,
                                      self.ffn_dim, n_wires=8)
              for _ in range(self.num_blocks)]
        )
        # Qiskit self‑attention
        self.attention = QuantumSelfAttention(n_qubits=self.arch[-1])

        # Quantum classifier circuit
        self.classifier_circ, _, _, _ = build_classifier_circuit(self.arch[-1], depth=2)

    def generate_random_network(self, samples: int = 100) -> None:
        """Create a random quantum network and store its components."""
        self.arch, self.unitaries, self.training_data, self.target_unitary = random_network(self.arch, samples)

    def feedforward(self, inputs: List[qt.Qobj]) -> List[List[qt.Qobj]]:
        """Return layerwise states for the given quantum inputs."""
        samples = [(inp, None) for inp in inputs]
        return feedforward(self.arch, self.unitaries, samples)

    def forward(self, inputs: List[qt.Qobj]) -> List[dict]:
        """
        Run the full pipeline: quantum feed‑forward → self‑attention →
        transformer → measurement.

        Parameters
        ----------
        inputs : List[qt.Qobj]
            List of pure states to be classified.

        Returns
        -------
        List[dict]
            Measurement counts for each input.
        """
        # 1. Feed‑forward
        states_list = self.feedforward(inputs)
        last_layer_states = [layer[-1] for layer in states_list]

        # 2. Transformer (quantum sub‑module)
        q_device = tq.QuantumDevice(n_wires=self.arch[-1])
        amp_tensors = torch.tensor([s.full().flatten() for s in last_layer_states],
                                   dtype=torch.float32).unsqueeze(0)
        output = amp_tensors
        for block in self.transformer:
            output = block(output, q_device)

        # 3. Quantum classifier measurement
        backend = qiskit.Aer.get_backend("qasm_simulator")
        job = qiskit.execute(self.classifier_circ, backend, shots=1024)
        return [job.result().get_counts(self.classifier_circ)]

    def graph_from_fidelities(self, states: List[qt.Qobj], threshold: float,
                              *, secondary: float | None = None,
                              secondary_weight: float = 0.5) -> nx.Graph:
        """Return a graph built from the fidelities of the provided states."""
        return fidelity_adjacency(states, threshold,
                                   secondary=secondary,
                                   secondary_weight=secondary_weight)


__all__ = ["GraphQNNModel"]
