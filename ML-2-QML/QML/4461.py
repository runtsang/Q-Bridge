"""Hybrid quantum kernel method mirroring the classical interface but using quantum circuits."""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import networkx as nx
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
import qiskit_machine_learning.neural_networks as qml_nn
import qiskit.primitives as primitives
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict

Tensor = np.ndarray

class HybridKernelMethod:
    """Quantum counterpart of HybridKernelMethod. Implements quantum kernel evaluation, 
    quantum graph neural network propagation, and quantum classifier/sampler construction."""

    # --- Quantum kernel ------------------------------------------------------
    class KernalAnsatz(tq.QuantumModule):
        """Encodes classical data into a quantum device with a fixed gate list."""
        def __init__(self, func_list):
            super().__init__()
            self.func_list = func_list

        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice, x: Tensor, y: Tensor) -> None:
            q_device.reset_states(x.shape[0])
            for info in self.func_list:
                params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
                func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
            for info in reversed(self.func_list):
                params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
                func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

    class Kernel(tq.QuantumModule):
        """Quantum kernel evaluated via a fixed TorchQuantum ansatz."""
        def __init__(self) -> None:
            super().__init__()
            self.n_wires = 4
            self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
            self.ansatz = HybridKernelMethod.KernalAnsatz(
                [
                    {"input_idx": [0], "func": "ry", "wires": [0]},
                    {"input_idx": [1], "func": "ry", "wires": [1]},
                    {"input_idx": [2], "func": "ry", "wires": [2]},
                    {"input_idx": [3], "func": "ry", "wires": [3]},
                ]
            )

        def forward(self, x: Tensor, y: Tensor) -> Tensor:
            x = x.reshape(1, -1)
            y = y.reshape(1, -1)
            self.ansatz(self.q_device, x, y)
            return torch.abs(self.q_device.states.view(-1)[0])

    @staticmethod
    def kernel_matrix(a: Sequence[Tensor], b: Sequence[Tensor]) -> np.ndarray:
        """Evaluate the Gram matrix between datasets a and b using the quantum kernel."""
        kernel = HybridKernelMethod.Kernel()
        return np.array([[kernel(x, y).item() for y in b] for x in a])

    # --- Quantum graph neural network ----------------------------------------
    @staticmethod
    def _tensored_id(num_qubits: int):
        return qiskit.quantum_info.Operator.identity(num_qubits)

    @staticmethod
    def _tensored_zero(num_qubits: int):
        return qiskit.quantum_info.Operator.zero(num_qubits)

    @staticmethod
    def _swap_registers(op: qiskit.quantum_info.Operator, source: int, target: int) -> qiskit.quantum_info.Operator:
        if source == target:
            return op
        order = list(range(op.num_qubits))
        order[source], order[target] = order[target], order[source]
        return op.permute(order)

    @staticmethod
    def _random_qubit_unitary(num_qubits: int):
        dim = 2 ** num_qubits
        matrix = np.random.normal(size=(dim, dim)) + 1j * np.random.normal(size=(dim, dim))
        unitary = np.linalg.qr(matrix)[0]
        return qiskit.quantum_info.Operator(unitary)

    @staticmethod
    def _random_qubit_state(num_qubits: int):
        dim = 2 ** num_qubits
        amplitudes = np.random.normal(size=(dim, 1)) + 1j * np.random.normal(size=(dim, 1))
        amplitudes /= np.linalg.norm(amplitudes)
        return qiskit.quantum_info.Operator(amplitudes)

    @staticmethod
    def random_training_data(unitary: qiskit.quantum_info.Operator, samples: int):
        dataset = []
        num_qubits = unitary.num_qubits
        for _ in range(samples):
            state = HybridKernelMethod._random_qubit_state(num_qubits)
            dataset.append((state, unitary @ state))
        return dataset

    @staticmethod
    def random_network(qnn_arch: List[int], samples: int):
        target_unitary = HybridKernelMethod._random_qubit_unitary(qnn_arch[-1])
        training_data = HybridKernelMethod.random_training_data(target_unitary, samples)
        unitaries: List[List[qiskit.quantum_info.Operator]] = [[]]
        for layer in range(1, len(qnn_arch)):
            num_inputs = qnn_arch[layer - 1]
            num_outputs = qnn_arch[layer]
            layer_ops: List[qiskit.quantum_info.Operator] = []
            for output in range(num_outputs):
                op = HybridKernelMethod._random_qubit_unitary(num_inputs + 1)
                if num_outputs > 1:
                    op = qiskit.quantum_info.Operator.tensor(
                        HybridKernelMethod._random_qubit_unitary(num_inputs + 1),
                        HybridKernelMethod._tensored_id(num_outputs - 1)
                    )
                    op = HybridKernelMethod._swap_registers(op, num_inputs, num_inputs + output)
                layer_ops.append(op)
            unitaries.append(layer_ops)
        return qnn_arch, unitaries, training_data, target_unitary

    @staticmethod
    def _layer_channel(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qiskit.quantum_info.Operator]], layer: int, input_state: qiskit.quantum_info.Operator):
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        state = qiskit.quantum_info.Operator.tensor(input_state, HybridKernelMethod._tensored_zero(num_outputs))
        layer_unitary = unitaries[layer][0]
        for gate in unitaries[layer][1:]:
            layer_unitary = gate @ layer_unitary
        return (layer_unitary @ state @ layer_unitary.dag()).ptrace(list(range(num_inputs)))

    @staticmethod
    def feedforward(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qiskit.quantum_info.Operator]], samples: Iterable[tuple[qiskit.quantum_info.Operator, qiskit.quantum_info.Operator]]):
        stored_states = []
        for sample, _ in samples:
            layerwise = [sample]
            current_state = sample
            for layer in range(1, len(qnn_arch)):
                current_state = HybridKernelMethod._layer_channel(qnn_arch, unitaries, layer, current_state)
                layerwise.append(current_state)
            stored_states.append(layerwise)
        return stored_states

    @staticmethod
    def state_fidelity(a: qiskit.quantum_info.Operator, b: qiskit.quantum_info.Operator) -> float:
        return abs((a.dag() @ b).data[0, 0]) ** 2

    @staticmethod
    def fidelity_adjacency(states: Sequence[qiskit.quantum_info.Operator], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5):
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = HybridKernelMethod.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    # --- Quantum classifier utilities -----------------------------------------
    @staticmethod
    def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
        encoding = ParameterVector("x", num_qubits)
        weights = ParameterVector("theta", num_qubits * depth)
        circuit = QuantumCircuit(num_qubits)
        for param, qubit in zip(encoding, range(num_qubits)):
            circuit.rx(param, qubit)
        index = 0
        for _ in range(depth):
            for qubit in range(num_qubits):
                circuit.ry(weights[index], qubit)
                index += 1
            for qubit in range(num_qubits - 1):
                circuit.cz(qubit, qubit + 1)
        observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
        return circuit, list(encoding), list(weights), observables

    # --- Quantum sampler utilities --------------------------------------------
    @staticmethod
    def SamplerQNN():
        """Return a Qiskit SamplerQNN instance."""
        inputs2 = ParameterVector("input", 2)
        weights2 = ParameterVector("weight", 4)
        qc2 = QuantumCircuit(2)
        qc2.ry(inputs2[0], 0)
        qc2.ry(inputs2[1], 1)
        qc2.cx(0, 1)
        qc2.ry(weights2[0], 0)
        qc2.ry(weights2[1], 1)
        qc2.cx(0, 1)
        qc2.ry(weights2[2], 0)
        qc2.ry(weights2[3], 1)
        sampler = primitives.StatevectorSampler()
        sampler_qnn = qml_nn.SamplerQNN(circuit=qc2, input_params=inputs2, weight_params=weights2, sampler=sampler)
        return sampler_qnn

    # --- Graph from quantum kernel --------------------------------------------
    @staticmethod
    def kernel_graph(a: Sequence[Tensor], threshold: float) -> nx.Graph:
        mat = HybridKernelMethod.kernel_matrix(a, a)
        graph = nx.Graph()
        graph.add_nodes_from(range(len(a)))
        for i in range(len(a)):
            for j in range(i + 1, len(a)):
                weight = mat[i, j]
                if weight >= threshold:
                    graph.add_edge(i, j, weight=weight)
        return graph

    __all__ = ["HybridKernelMethod"]
