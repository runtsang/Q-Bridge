import qiskit
import numpy as np
import networkx as nx
import torch
import torchquantum as tq
from typing import List, Sequence, Tuple, Iterable

# ----------------------------------------------------------------------
# 1. Quantum kernel using TorchQuantum (inspired by QuantumKernelMethod.py)
# ----------------------------------------------------------------------
class QuantumKernel(tq.QuantumModule):
    """Variational kernel that encodes two input vectors via Ry gates."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = tq.GeneralEncoder(
            [{"input_idx": [0], "func": "ry", "wires": [0]},
             {"input_idx": [1], "func": "ry", "wires": [1]},
             {"input_idx": [2], "func": "ry", "wires": [2]},
             {"input_idx": [3], "func": "ry", "wires": [3]}]
        )
        self.kernel_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x)
        self.kernel_layer(self.q_device)
        out = self.measure(self.q_device)
        return torch.abs(out[0])


# ----------------------------------------------------------------------
# 2. Quantum GraphQNN (inspired by GraphQNN.py + Qiskit circuits)
# ----------------------------------------------------------------------
class GraphQNNQuantum:
    """
    Quantum‑enhanced Graph Neural Network.
    Provides the same API as the classical version but uses Qiskit circuits
    for feature propagation and TorchQuantum for kernel evaluation.
    """

    def __init__(self,
                 arch: Sequence[int],
                 kernel_type: str = "quantum",
                 conv_type: str = "quanv",
                 backend: str = "qasm_simulator",
                 shots: int = 1024):
        self.arch = list(arch)
        self.kernel_type = kernel_type
        self.conv_type = conv_type
        self.backend = qiskit.Aer.get_backend(backend)
        self.shots = shots
        self.unitaries: List[List[qiskit.QuantumCircuit]] = []
        self._build_layers()

    def _build_layers(self) -> None:
        for in_f, out_f in zip(self.arch[:-1], self.arch[1:]):
            layer: List[qiskit.QuantumCircuit] = []
            for _ in range(out_f):
                qc = qiskit.QuantumCircuit(in_f + 1)
                # Encode inputs as RX rotations
                for i in range(in_f):
                    qc.rx(qiskit.circuit.Parameter(f"theta{i}"), i)
                # Random two‑qubit operations
                qc += qiskit.circuit.random.random_circuit(in_f + 1, 2)
                qc.measure_all()
                layer.append(qc)
            self.unitaries.append(layer)

    def random_network(self, samples: int = 100) -> Tuple[List[int], List[List[qiskit.QuantumCircuit]], List[Tuple[np.ndarray, dict]], qiskit.QuantumCircuit]:
        target_qc = qiskit.QuantumCircuit(self.arch[-1])
        target_qc += qiskit.circuit.random.random_circuit(self.arch[-1], 2)
        target_qc.measure_all()
        data: List[Tuple[np.ndarray, dict]] = []
        for _ in range(samples):
            inp = np.random.rand(self.arch[0])
            job = qiskit.execute(
                target_qc,
                self.backend,
                shots=self.shots,
                parameter_binds=[{f"theta{i}": np.pi if v > 0.5 else 0 for i, v in enumerate(inp)}]
            )
            result = job.result().get_counts(target_qc)
            data.append((inp, result))
        return self.arch, self.unitaries, data, target_qc

    def _run_circuit(self, qc: qiskit.QuantumCircuit, data: np.ndarray) -> float:
        param_binds = [{f"theta{i}": np.pi if val > 0.5 else 0 for i, val in enumerate(data)}]
        job = qiskit.execute(qc, self.backend, shots=self.shots, parameter_binds=param_binds)
        result = job.result().get_counts(qc)
        total = sum(int(bits.count("1")) * cnt for bits, cnt in result.items())
        return total / (self.shots * len(data))

    def feedforward(self,
                    samples: Iterable[Tuple[np.ndarray, dict]]) -> List[List[np.ndarray]]:
        states: List[List[np.ndarray]] = []
        for x, _ in samples:
            layerwise: List[np.ndarray] = [x]
            current = x
            for layer in self.unitaries:
                outputs = []
                for qc in layer:
                    prob = self._run_circuit(qc, current)
                    outputs.append(prob)
                current = np.array(outputs)
                layerwise.append(current)
            states.append(layerwise)
        return states

    def state_fidelity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float((np.linalg.norm(a) * np.linalg.norm(b)) ** -2 * np.dot(a, b) ** 2)

    def fidelity_adjacency(self,
                           states: Sequence[np.ndarray],
                           threshold: float,
                           *,
                           secondary: float | None = None,
                           secondary_weight: float = 0.5) -> nx.Graph:
        G = nx.Graph()
        G.add_nodes_from(range(len(states)))
        for i, a in enumerate(states):
            for j, b in enumerate(states[i + 1:], start=i + 1):
                fid = self.state_fidelity(a, b)
                if fid >= threshold:
                    G.add_edge(i, j, weight=1.0)
                elif secondary is not None and fid >= secondary:
                    G.add_edge(i, j, weight=secondary_weight)
        return G

    def kernel_matrix(self,
                      a: Sequence[np.ndarray],
                      b: Sequence[np.ndarray]) -> np.ndarray:
        if self.kernel_type == "quantum":
            kernel = QuantumKernel()
            return np.array([[kernel(torch.from_numpy(x), torch.from_numpy(y)).item()
                              for y in b] for x in a])
        else:
            return np.zeros((len(a), len(b)))

__all__ = ["GraphQNNQuantum", "QuantumKernel"]
