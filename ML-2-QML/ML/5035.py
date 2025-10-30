"""GraphQNNGen355: hybrid graph neural‑network utilities.

The class can operate in classical or quantum mode, providing a unified API for
network generation, forward propagation, fidelity calculations, and graph
construction.  It also embeds a simple fully‑connected layer and a
classifier‑circuit factory for cross‑validation experiments."""
from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple, Union

import networkx as nx
import torch
import torch.nn as nn
import numpy as np

Tensor = torch.Tensor
Qobj = object  # placeholder for quantum object type in the classical module


class GraphQNNGen355:
    """
    Hybrid graph‑neural‑network helper.

    * Classical mode uses dense linear layers and PyTorch tensors.
    * Quantum mode builds a Qiskit circuit and evaluates it on a simulator.
    The public API is identical across modes, making side‑by‑side benchmarking
    trivial.
    """

    def __init__(self, qnn_arch: Sequence[int], mode: str = "classical") -> None:
        self.arch = list(qnn_arch)
        self.mode = mode
        self._validate_mode()
        if mode == "classical":
            self._build_classical()
        else:
            self._build_quantum()

    # --------------------------------------------------------------------------
    # mode validation
    # --------------------------------------------------------------------------
    def _validate_mode(self) -> None:
        if self.mode not in {"classical", "quantum"}:
            raise ValueError("mode must be either 'classical' or 'quantum'")

    # --------------------------------------------------------------------------
    # classical helpers
    # --------------------------------------------------------------------------
    def _build_classical(self) -> None:
        self.weights: List[Tuple[Tensor, Tensor]] = []
        for in_f, out_f in zip(self.arch[:-1], self.arch[1:]):
            w = nn.Parameter(torch.randn(out_f, in_f))
            b = nn.Parameter(torch.randn(out_f))
            self.weights.append((w, b))

    def random_network_classical(self, samples: int) -> Tuple[List[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
        weights: List[Tensor] = []
        for in_f, out_f in zip(self.arch[:-1], self.arch[1:]):
            w = torch.randn(out_f, in_f)
            weights.append(w)
        target_weight = weights[-1]
        training_data = self._random_training_data(target_weight, samples)
        return self.arch, weights, training_data, target_weight

    @staticmethod
    def _random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
        dataset: List[Tuple[Tensor, Tensor]] = []
        for _ in range(samples):
            features = torch.randn(weight.size(1))
            target = weight @ features
            dataset.append((features, target))
        return dataset

    def feedforward_classical(self, samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
        stored: List[List[Tensor]] = []
        for features, _ in samples:
            activations = [features]
            current = features
            for w, b in self.weights:
                current = torch.tanh(w @ current + b)
                activations.append(current)
            stored.append(activations)
        return stored

    @staticmethod
    def state_fidelity_classical(a: Tensor, b: Tensor) -> float:
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float((a_norm @ b_norm).item() ** 2)

    def fidelity_adjacency_classical(
        self,
        states: Sequence[Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
            fid = self.state_fidelity_classical(s_i, s_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    # --------------------------------------------------------------------------
    # quantum helpers
    # --------------------------------------------------------------------------
    def _build_quantum(self) -> None:
        import qiskit
        from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
        from qiskit.circuit import ParameterVector
        from qiskit.quantum_info import Statevector, SparsePauliOp

        self.qiskit = qiskit
        self._QuantumCircuit = QuantumCircuit
        self._QuantumRegister = QuantumRegister
        self._ClassicalRegister = ClassicalRegister
        self._execute = execute
        self._ParameterVector = ParameterVector
        self._Statevector = Statevector
        self._SparsePauliOp = SparsePauliOp
        self.backend = self.qiskit.Aer.get_backend("qasm_simulator")
        self.shots = 1024
        self.unitaries: List[List[QuantumCircuit]] = [[]]

    def _random_unitary(self, num_qubits: int) -> np.ndarray:
        dim = 2 ** num_qubits
        matrix = np.random.normal(size=(dim, dim)) + 1j * np.random.normal(size=(dim, dim))
        q, _ = np.linalg.qr(matrix)
        return q

    def random_network_quantum(self, samples: int) -> Tuple[List[int], List[List[QuantumCircuit]], List[Tuple[Statevector, Statevector]], Statevector]:
        target_unitary = self._random_unitary(self.arch[-1])
        training_data = [
            (
                self._Statevector.from_label("0" * self.arch[-1]),
                self._Statevector(target_unitary @ self._Statevector.from_label("0" * self.arch[-1]).data),
            )
            for _ in range(samples)
        ]
        unitaries: List[List[QuantumCircuit]] = [[]]
        for layer in range(1, len(self.arch)):
            num_inputs = self.arch[layer - 1]
            num_outputs = self.arch[layer]
            layer_ops: List[QuantumCircuit] = []
            for output in range(num_outputs):
                qc = self._QuantumCircuit(num_inputs + 1)
                qc.h(range(num_inputs + 1))
                qc.barrier()
                qc.ry(self._ParameterVector(f"theta_{layer}_{output}", 1), num_inputs + output)
                layer_ops.append(qc)
            unitaries.append(layer_ops)
        return self.arch, unitaries, training_data, target_unitary

    def feedforward_quantum(self, samples: Iterable[Tuple[Statevector, Statevector]]) -> List[List[Statevector]]:
        stored: List[List[Statevector]] = []
        for state, _ in samples:
            layerwise = [state]
            for layer in range(1, len(self.arch)):
                for gate in self.unitaries[layer]:
                    self._execute(gate, self.backend, shots=self.shots)
                state_vec = self._Statevector.from_label("0" * self.arch[layer])
                layerwise.append(state_vec)
            stored.append(layerwise)
        return stored

    def state_fidelity_quantum(self, a: Statevector, b: Statevector) -> float:
        return abs((a.data.conj() @ b.data)) ** 2

    def fidelity_adjacency_quantum(
        self,
        states: Sequence[Statevector],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
            fid = self.state_fidelity_quantum(s_i, s_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    # --------------------------------------------------------------------------
    # hybrid utilities
    # --------------------------------------------------------------------------
    def convert_classical_weights_to_unitary(self, weight: Tensor) -> np.ndarray:
        w_np = weight.detach().cpu().numpy()
        u, _, vh = np.linalg.svd(w_np, full_matrices=False)
        return u @ vh

    def fully_connected_layer(self, thetas: Iterable[float]) -> np.ndarray:
        thetas_t = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        linear = nn.Linear(thetas_t.size(0), 1)
        return torch.tanh(linear(thetas_t)).mean().detach().numpy()

    def build_classifier_circuit(self, num_qubits: int, depth: int) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
        encoding = self._ParameterVector("x", num_qubits)
        weights = self._ParameterVector("theta", num_qubits * depth)
        qc = self._QuantumCircuit(num_qubits)
        for param, qubit in zip(encoding, range(num_qubits)):
            qc.rx(param, qubit)
        idx = 0
        for _ in range(depth):
            for qubit in range(num_qubits):
                qc.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(num_qubits - 1):
                qc.cz(qubit, qubit + 1)
        observables = [self._SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
        return qc, list(encoding), list(weights), observables

    # --------------------------------------------------------------------------
    # public API wrappers
    # --------------------------------------------------------------------------
    def random_network(self, samples: int) -> Tuple[List[int], List[Union[List[Tensor], List[QuantumCircuit]]], List[Tuple[Union[Tensor, Statevector], Union[Tensor, Statevector]]], Union[Tensor, Statevector]]:
        if self.mode == "classical":
            return self.random_network_classical(samples)
        return self.random_network_quantum(samples)

    def feedforward(self, samples: Iterable[Tuple[Union[Tensor, Statevector], Union[Tensor, Statevector]]]) -> List[List[Union[Tensor, Statevector]]]:
        if self.mode == "classical":
            return self.feedforward_classical(samples)
        return self.feedforward_quantum(samples)

    def fidelity_adjacency(self, states: Sequence[Union[Tensor, Statevector]], threshold: float,
                           *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
        if self.mode == "classical":
            return self.fidelity_adjacency_classical(states, threshold, secondary=secondary, secondary_weight=secondary_weight)
        return self.fidelity_adjacency_quantum(states, threshold, secondary=secondary, secondary_weight=secondary_weight)

    def state_fidelity(self, a: Union[Tensor, Statevector], b: Union[Tensor, Statevector]) -> float:
        if self.mode == "classical":
            return self.state_fidelity_classical(a, b)
        return self.state_fidelity_quantum(a, b)


__all__ = ["GraphQNNGen355"]
