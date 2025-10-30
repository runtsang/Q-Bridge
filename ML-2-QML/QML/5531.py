from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple, Callable

import networkx as nx
import numpy as np
import torch
import torchquantum as tq
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

# ------------------------------------------------------------------
# Quantum kernel ansatz
# ------------------------------------------------------------------
class KernalAnsatz(tq.QuantumModule):
    """Programmable list of gates that encodes two classical vectors."""
    def __init__(self, func_list: List[dict]) -> None:
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

class Kernel(tq.QuantumModule):
    """Fixed quantum kernel built from a simple Ry encoding."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = KernalAnsatz(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

# ------------------------------------------------------------------
# Quantum quanvolution primitives
# ------------------------------------------------------------------
class QuantumQuanvolutionFilter(tq.QuantumModule):
    """Two‑qubit kernel applied to 2×2 patches extracted from a 28×28 image."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = tq.QuantumDevice(self.n_wires, bsz=bsz, device=x.device)
        x = x.view(bsz, 28, 28)
        patches: List[torch.Tensor] = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                data = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(device, data)
                self.layer(device)
                measurement = self.measure(device)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)

class QuantumQuanvolutionClassifier(tq.QuantumModule):
    """Hybrid classifier that wraps the quantum filter and a classical linear head."""
    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuantumQuanvolutionFilter()
        self.linear = tq.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return tq.functional.log_softmax(logits, dim=-1)

# ------------------------------------------------------------------
# Fast estimator for quantum circuits
# ------------------------------------------------------------------
class FastBaseEstimator:
    """Expectation‑value calculator for a parameterised circuit."""
    def __init__(self, circuit: QuantumCircuit) -> None:
        self.circuit = circuit
        self.parameters = list(circuit.parameters)

    def _bind(self, param_vals: Sequence[float]) -> QuantumCircuit:
        if len(param_vals)!= len(self.parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self.parameters, param_vals))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        results: List[List[complex]] = []
        for vals in parameter_sets:
            bound = self._bind(vals)
            state = Statevector.from_instruction(bound)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

# ------------------------------------------------------------------
# Quantum graph‑neural‑network hybrid
# ------------------------------------------------------------------
class GraphQNNHybrid:
    """
    Quantum implementation of GraphQNNHybrid that:
      * builds a random quantum network of unitary layers,
      * propagates states through the network,
      * constructs a fidelity‑based adjacency graph,
      * offers a fast estimator based on a qiskit circuit.
    """
    def __init__(self,
                 arch: Sequence[int],
                 random_seed: int | None = None) -> None:
        self.arch = list(arch)
        self.random_seed = random_seed
        self.unitaries: List[List[tq.Qobj]] = []
        self._build_random_network()
        # Simple parameterised circuit for FastEstimator
        self.circuit = QuantumCircuit(self.arch[-1])
        self.parameters: List[Parameter] = []
        for layer in range(1, len(self.arch)):
            for q in range(self.arch[layer - 1]):
                p = Parameter(f"theta_{layer}_{q}")
                self.parameters.append(p)
                self.circuit.ry(p, q)

    # ------------------------------------------------------------------
    # Random network construction
    # ------------------------------------------------------------------
    def _build_random_network(self) -> None:
        rng = np.random.default_rng(self.random_seed)
        for layer in range(1, len(self.arch)):
            num_inputs = self.arch[layer - 1]
            num_outputs = self.arch[layer]
            layer_ops: List[tq.Qobj] = []
            for _ in range(num_outputs):
                op = tq.random_unitary(num_inputs + 1)
                if num_outputs > 1:
                    op = tq.tensor(op, tq.identity(num_outputs - 1))
                    op = tq.swap(op, num_inputs, num_inputs + _)
                layer_ops.append(op)
            self.unitaries.append(layer_ops)

    # ------------------------------------------------------------------
    # Partial trace helper
    # ------------------------------------------------------------------
    def _partial_trace(self, state: tq.Qobj, keep: Sequence[int]) -> tq.Qobj:
        return state.ptrace(list(keep))

    # ------------------------------------------------------------------
    # Forward propagation
    # ------------------------------------------------------------------
    def feedforward(
        self,
        samples: Iterable[Tuple[tq.Qobj, tq.Qobj]],
    ) -> List[List[tq.Qobj]]:
        outputs: List[List[tq.Qobj]] = []
        for sample, _ in samples:
            layerwise = [sample]
            current = sample
            for layer_idx, ops in enumerate(self.unitaries, start=1):
                state = tq.tensor(current, tq.zero(self.arch[layer_idx]))
                unitary = ops[0]
                for gate in ops[1:]:
                    unitary = gate @ unitary
                current = self._partial_trace(unitary @ state @ unitary.dag(), range(self.arch[layer_idx - 1]))
                layerwise.append(current)
            outputs.append(layerwise)
        return outputs

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------
    def fidelity_adjacency(
        self,
        states: Sequence[tq.Qobj],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, si), (j, sj) in itertools.combinations(enumerate(states), 2):
            fid = abs((si.dag() @ sj)[0, 0]) ** 2
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    # ------------------------------------------------------------------
    # Evaluation routine
    # ------------------------------------------------------------------
    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        estimator = FastBaseEstimator(self.circuit)
        return estimator.evaluate(observables, parameter_sets)

    # ------------------------------------------------------------------
    # Synthetic training data generator
    # ------------------------------------------------------------------
    def random_training_data(
        self,
        target_unitary: tq.Qobj,
        samples: int,
    ) -> List[Tuple[tq.Qobj, tq.Qobj]]:
        data: List[Tuple[tq.Qobj, tq.Qobj]] = []
        num_qubits = len(target_unitary.dims[0])
        for _ in range(samples):
            state = tq.random_state(num_qubits)
            data.append((state, target_unitary * state))
        return data

    # ------------------------------------------------------------------
    # Convenience static helper
    # ------------------------------------------------------------------
    @staticmethod
    def random_network(
        arch: Sequence[int],
        samples: int,
        random_seed: int | None = None,
    ) -> Tuple[List[int], List[List[tq.Qobj]], List[Tuple[tq.Qobj, tq.Qobj]], tq.Qobj]:
        rng = np.random.default_rng(random_seed)
        unitaries: List[List[tq.Qobj]] = []
        for layer in range(1, len(arch)):
            num_inputs = arch[layer - 1]
            num_outputs = arch[layer]
            layer_ops: List[tq.Qobj] = []
            for _ in range(num_outputs):
                op = tq.random_unitary(num_inputs + 1)
                if num_outputs > 1:
                    op = tq.tensor(op, tq.identity(num_outputs - 1))
                    op = tq.swap(op, num_inputs, num_inputs + _)
                layer_ops.append(op)
            unitaries.append(layer_ops)
        target_unitary = tq.random_unitary(arch[-1])
        training_data = GraphQNNHybrid.random_training_data(target_unitary, samples)
        return list(arch), unitaries, training_data, target_unitary

__all__ = [
    "KernalAnsatz",
    "Kernel",
    "QuantumQuanvolutionFilter",
    "QuantumQuanvolutionClassifier",
    "FastBaseEstimator",
    "GraphQNNHybrid",
]
