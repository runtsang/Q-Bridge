"""Hybrid fully‑connected layer with classical, quantum, and graph components.

This module implements a hybrid class that combines a classical fully‑connected
layer, a simple quantum circuit (variational circuit implemented with qiskit),
and a graph‑based fidelity adjacency.  It mirrors the original ``FCL`` interface
but adds a quantum‑parameterized circuit to the classical feed‑forward path.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
import networkx as nx
import itertools
from typing import Iterable, Sequence, List, Tuple, Optional

import qiskit
import qutip as qt

def _tensored_id(num_qubits: int) -> qt.Qobj:
    identity = qt.qeye(2 ** num_qubits)
    dims = [2] * num_qubits
    identity.dims = [dims.copy(), dims.copy()]
    return identity

def _swap_registers(op: qt.Qobj, source: int, target: int) -> qt.Qobj:
    if source == target:
        return op
    order = list(range(len(op.dims[0])))
    order[source], order[target] = order[target], order[source]
    return op.permute(order)

def _random_qubit_unitary(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    matrix = np.random.normal(size=(dim, dim)) + 1j * np.random.normal(size=(dim, dim))
    unitary = np.linalg.qr(matrix)[0]
    qobj = qt.Qobj(unitary)
    dims = [2] * num_qubits
    qobj.dims = [dims.copy(), dims.copy()]
    return qobj

def _random_qubit_state(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    amplitudes = np.random.normal(size=(dim, 1)) + 1j * np.random.normal(size=(dim, 1))
    amplitudes /= np.linalg.norm(amplitudes)
    state = qt.Qobj(amplitudes)
    state.dims = [[2] * num_qubits, [1] * num_qubits]
    return state

def random_training_data(unitary: qt.Qobj, samples: int = 10) -> List[Tuple[qt.Qobj, qt.Qobj]]:
    dataset: List[Tuple[qt.Qobj, qt.Qobj]] = []
    num_qubits = len(unitary.dims[0])
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        dataset.append((state, unitary * state))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int = 10) -> Tuple[List[int], List[List[qt.Qobj]], List[Tuple[qt.Qobj, qt.Qobj]], qt.Qobj]:
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

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

    return list(qnn_arch), unitaries, training_data, target_unitary

class HybridFullyConnectedGraphLayer(nn.Module):
    def __init__(self, n_features: int = 1, n_qubits: int = 1, shots: int = 100,
                 backend: Optional[qiskit.providers.Provider] = None) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.qc = self._build_circuit()

    def _build_circuit(self) -> qiskit.QuantumCircuit:
        qc = qiskit.QuantumCircuit(self.n_qubits)
        theta = qiskit.circuit.Parameter("theta")
        qc.h(range(self.n_qubits))
        qc.barrier()
        qc.ry(theta, range(self.n_qubits))
        qc.measure_all()
        return qc

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        torch_thetas = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        classical_out = torch.tanh(self.linear(torch_thetas)).mean(dim=0).item()
        expectation = 0.0
        for theta in thetas:
            job = qiskit.execute(
                self.qc,
                self.backend,
                shots=self.shots,
                parameter_binds=[{self.qc.parameters[0]: theta}],
            )
            result = job.result()
            counts = result.get_counts(self.qc)
            probs = np.array(list(counts.values())) / self.shots
            states = np.array([int(k, 2) for k in counts.keys()], dtype=float)
            expectation += np.sum(states * probs)
        expectation /= len(thetas)
        return np.array([classical_out, expectation])

    def feedforward(self, samples: Iterable[torch.Tensor]) -> List[List[torch.Tensor]]:
        activations_per_sample: List[List[torch.Tensor]] = []
        for sample in samples:
            activations = [sample]
            current = torch.tanh(self.linear(sample))
            activations.append(current)
            activations_per_sample.append(activations)
        return activations_per_sample

    @staticmethod
    def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
        return float(abs((a.dag() * b)[0, 0]) ** 2)

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[qt.Qobj],
        threshold: float,
        *,
        secondary: Optional[float] = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = HybridFullyConnectedGraphLayer.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

def FCL(n_features: int = 1, n_qubits: int = 1, shots: int = 100,
        backend: Optional[qiskit.providers.Provider] = None) -> HybridFullyConnectedGraphLayer:
    return HybridFullyConnectedGraphLayer(n_features, n_qubits, shots, backend)

__all__ = ["HybridFullyConnectedGraphLayer", "FCL"]
