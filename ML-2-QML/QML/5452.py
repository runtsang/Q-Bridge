import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torchquantum.functional import func_name_dict
import qutip as qt
import networkx as nx
import itertools
from typing import Iterable, Sequence, Tuple, List

# --------------------------------------------------------------------------- #
#  Quantum kernel (TorchQuantum)
# --------------------------------------------------------------------------- #
class QuantumRBF(tq.QuantumModule):
    """Quantum kernel that encodes two feature vectors into a shared device."""
    def __init__(self, n_wires: int = 4, gate_list: Iterable[dict] | None = None):
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        # Default encoding: Ry on each wire
        self.gate_list = gate_list or [
            {"input_idx": [0], "func": "ry", "wires": [0]},
            {"input_idx": [1], "func": "ry", "wires": [1]},
            {"input_idx": [2], "func": "ry", "wires": [2]},
            {"input_idx": [3], "func": "ry", "wires": [3]},
        ]

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        # Encode x
        for info in self.gate_list:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        # Un‑encode y (with negative parameters)
        for info in reversed(self.gate_list):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

    def forward_overlap(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.forward(self.q_device, x, y)
        # Overlap is the absolute value of the first state amplitude
        return torch.abs(self.q_device.states.view(-1)[0])

def quantum_kernel_matrix(
    a: Sequence[torch.Tensor],
    b: Sequence[torch.Tensor],
    n_wires: int = 4
) -> np.ndarray:
    kernel = QuantumRBF(n_wires)
    return np.array([[kernel.forward_overlap(x, y).item() for y in b] for x in a])

# --------------------------------------------------------------------------- #
#  Quantum graph utilities (qutip)
# --------------------------------------------------------------------------- #
def _tensored_id(num_qubits: int) -> qt.Qobj:
    iden = qt.qeye(2 ** num_qubits)
    dims = [2] * num_qubits
    iden.dims = [dims.copy(), dims.copy()]
    return iden

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
    mat = np.random.normal(size=(dim, dim)) + 1j * np.random.normal(size=(dim, dim))
    u, _ = np.linalg.qr(mat)
    qobj = qt.Qobj(u)
    dims = [2] * num_qubits
    qobj.dims = [dims.copy(), dims.copy()]
    return qobj

def _random_qubit_state(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    amp = np.random.normal(size=(dim, 1)) + 1j * np.random.normal(size=(dim, 1))
    amp /= np.linalg.norm(amp)
    state = qt.Qobj(amp)
    state.dims = [[2] * num_qubits, [1] * num_qubits]
    return state

def random_network(
    qnn_arch: List[int],
    samples: int
) -> Tuple[List[int], List[List[qt.Qobj]], List[Tuple[qt.Qobj, qt.Qobj]], qt.Qobj]:
    """Generate a random layered quantum network and synthetic training data."""
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = [( _random_qubit_state(len(target_unitary.dims[0])), target_unitary * _random_qubit_state(len(target_unitary.dims[0])) ) for _ in range(samples)]

    unitaries: List[List[qt.Qobj]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        layer_ops: List[qt.Qobj] = []
        for output in range(num_outputs):
            op = _random_qubit_unitary(num_inputs + 1)
            if num_outputs > 1:
                op = qt.tensor(_random_qubit_unitary(num_inputs + 1), _tensored_id(num_outputs - 1))
                op = _swap_registers(op, num_inputs, num_inputs + output)
            layer_ops.append(op)
        unitaries.append(layer_ops)

    return qnn_arch, unitaries, training_data, target_unitary

def _partial_trace_remove(state: qt.Qobj, remove: Sequence[int]) -> qt.Qobj:
    keep = list(range(len(state.dims[0])))
    for idx in sorted(remove, reverse=True):
        keep.pop(idx)
    return state.ptrace(keep)

def _layer_channel(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[qt.Qobj]],
    layer: int,
    input_state: qt.Qobj
) -> qt.Qobj:
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    state = qt.tensor(input_state, _tensored_zero(num_outputs))
    layer_unitary = unitaries[layer][0].copy()
    for gate in unitaries[layer][1:]:
        layer_unitary = gate * layer_unitary
    return _partial_trace_remove(layer_unitary * state * layer_unitary.dag(), range(num_inputs))

def feedforward(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[qt.Qobj]],
    samples: Iterable[Tuple[qt.Qobj, qt.Qobj]]
) -> List[List[qt.Qobj]]:
    """Forward pass of a layered quantum network."""
    stored = []
    for sample, _ in samples:
        layerwise = [sample]
        current = sample
        for layer in range(1, len(qnn_arch)):
            current = _layer_channel(qnn_arch, unitaries, layer, current)
            layerwise.append(current)
        stored.append(layerwise)
    return stored

def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
    return abs((a.dag() * b)[0, 0]) ** 2

def fidelity_adjacency(
    states: Sequence[qt.Qobj],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5
) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, si), (j, sj) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(si, sj)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

# --------------------------------------------------------------------------- #
#  Hybrid quantum head (qiskit)
# --------------------------------------------------------------------------- #
import qiskit
from qiskit import assemble, transpile
from qiskit.quantum_info import SparsePauliOp

class QuantumHybridHead(nn.Module):
    """Hybrid layer that forwards activations through a 2‑qubit Qiskit circuit."""
    def __init__(self, n_qubits: int = 2, shots: int = 200):
        super().__init__()
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = qiskit.Aer.get_backend("aer_simulator")
        # Build a simple parametric circuit
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        theta = qiskit.circuit.Parameter("theta")
        self.circuit.h(range(n_qubits))
        self.circuit.ry(theta, 0)
        self.circuit.measure_all()
        self.theta = theta

    def run(self, params: np.ndarray) -> np.ndarray:
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(compiled, shots=self.shots, parameter_binds=[{self.theta: p} for p in params])
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        # Expectation value of Z on first qubit
        exp = 0.0
        for bitstring, count in result.items():
            bit = int(bitstring[-1])  # least significant bit corresponds to qubit 0
            exp += ((-1) ** bit) * count
        return np.array([exp / self.shots])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is expected to be a 1‑D tensor of parameter values
        py_arr = x.detach().cpu().numpy()
        exp_vals = self.run(py_arr)
        probs = 0.5 * (1 + exp_vals)  # map from [-1,1] to [0,1]
        return torch.tensor(probs, dtype=torch.float32)

# --------------------------------------------------------------------------- #
#  UnifiedKernelClassifier (quantum version)
# --------------------------------------------------------------------------- #
class UnifiedKernelClassifier(nn.Module):
    """
    Quantum‑ready classifier.  Supports:
      * 'quantum' – uses QuantumRBF kernel and a dense backbone.
      * 'hybrid'  – uses QuantumHybridHead on top of a dense backbone.
    Classical mode is not available here; see the classical module.
    """
    def __init__(self, mode: str = 'quantum', **kwargs):
        super().__init__()
        self.mode = mode.lower()
        if self.mode == 'quantum':
            self.kernel = QuantumRBF(kwargs.get('n_wires', 4))
            self.backbone = nn.Sequential(
                nn.Linear(kwargs.get('input_dim', 10), kwargs.get('hidden_dim', 20)),
                nn.ReLU(),
                nn.Linear(20, 2)
            )
        elif self.mode == 'hybrid':
            self.head = QuantumHybridHead(kwargs.get('n_qubits', 2), kwargs.get('shots', 200))
        else:
            raise NotImplementedError("Only 'quantum' and 'hybrid' modes are implemented in the quantum module.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == 'quantum':
            return self.backbone(x)
        elif self.mode == 'hybrid':
            return self.head(x)
        else:
            raise RuntimeError("Unsupported mode.")
