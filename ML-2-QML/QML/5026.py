"""GraphQNNHybrid: quantum graph neural network with optional self‑attention
and a sampler interface."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple, Optional

import networkx as nx
import numpy as np
import qutip as qt
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_machine_learning.neural_networks import SamplerQNN as QiskitSamplerQNN
from qiskit.primitives import StatevectorSampler

Tensor = qt.Qobj

# ----- Fraud parameters ---------------------------------------

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

# ----- Self‑attention ----------------------------------------

class QuantumSelfAttention:
    """Quantum self‑attention block built from single‑qubit rotations and
    controlled‑X entangling gates."""
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build_circuit(self,
                       rotation_params: np.ndarray,
                       entangle_params: np.ndarray) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(self,
            backend,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            shots: int = 1024) -> dict:
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = qiskit.execute(circuit, backend, shots=shots)
        return job.result().get_counts(circuit)

# ----- Sampler ----------------------------------------

def SamplerQNN() -> QiskitSamplerQNN:
    """Return a Qiskit SamplerQNN instance for measurement‑based inference."""
    inputs = qiskit.circuit.ParameterVector("input", 2)
    weights = qiskit.circuit.ParameterVector("weight", 4)
    qc = QuantumCircuit(2)
    qc.ry(inputs[0], 0)
    qc.ry(inputs[1], 1)
    qc.cx(0, 1)
    qc.ry(weights[0], 0)
    qc.ry(weights[1], 1)
    qc.cx(0, 1)
    qc.ry(weights[2], 0)
    qc.ry(weights[3], 1)
    sampler = StatevectorSampler()
    return QiskitSamplerQNN(circuit=qc,
                            input_params=inputs,
                            weight_params=weights,
                            sampler=sampler)

# ----- GraphQNNHybrid ----------------------------------------

class GraphQNNHybrid:
    """Quantum graph neural network that mirrors the classical GraphQNNHybrid.
    It propagates a quantum state through a stack of random unitaries and
    can produce a fidelity‑based adjacency graph of intermediate states."""
    def __init__(self,
                 qnn_arch: Sequence[int],
                 use_attention: bool = True,
                 fraud_params: Optional[Tuple[FraudLayerParameters,
                                              Iterable[FraudLayerParameters]]] = None) -> None:
        self.arch: List[int] = list(qnn_arch)
        self.unitaries: List[List[qt.Qobj]] = []

        # Fraud‑detection layers are kept for compatibility but not applied
        self.fraud_params = fraud_params

        # Build quantum layers
        for layer in range(1, len(self.arch)):
            num_inputs = self.arch[layer - 1]
            num_outputs = self.arch[layer]
            ops: List[qt.Qobj] = [self._random_qubit_unitary(num_inputs + 1)
                                  for _ in range(num_outputs)]
            self.unitaries.append(ops)

        self.attention = QuantumSelfAttention(self.arch[-1]) if use_attention else None

    # -------------------------------------------------------------------------
    @staticmethod
    def _random_qubit_unitary(num_qubits: int) -> qt.Qobj:
        dim = 2 ** num_qubits
        matrix = np.random.normal(size=(dim, dim)) + 1j * np.random.normal(size=(dim, dim))
        unitary, _ = np.linalg.qr(matrix)
        qobj = qt.Qobj(unitary)
        qobj.dims = [[2] * num_qubits, [2] * num_qubits]
        return qobj

    @staticmethod
    def _random_qubit_state(num_qubits: int) -> qt.Qobj:
        dim = 2 ** num_qubits
        vec = np.random.normal(size=(dim, 1)) + 1j * np.random.normal(size=(dim, 1))
        vec /= np.linalg.norm(vec)
        qobj = qt.Qobj(vec)
        qobj.dims = [[2] * num_qubits, [1] * num_qubits]
        return qobj

    @staticmethod
    def random_training_data(unitary: qt.Qobj, samples: int) -> List[Tuple[qt.Qobj, qt.Qobj]]:
        dataset: List[Tuple[qt.Qobj, qt.Qobj]] = []
        num_qubits = len(unitary.dims[0])
        for _ in range(samples):
            state = GraphQNNHybrid._random_qubit_state(num_qubits)
            dataset.append((state, unitary * state))
        return dataset

    @staticmethod
    def random_network(qnn_arch: List[int], samples: int):
        target_unitary = GraphQNNHybrid._random_qubit_unitary(qnn_arch[-1])
        training_data = GraphQNNHybrid.random_training_data(target_unitary, samples)

        layers: List[List[qt.Qobj]] = [[]]
        for layer in range(1, len(qnn_arch)):
            num_inputs = qnn_arch[layer - 1]
            num_outputs = qnn_arch[layer]
            layer_ops: List[qt.Qobj] = [GraphQNNHybrid._random_qubit_unitary(num_inputs + 1)
                                        for _ in range(num_outputs)]
            layers.append(layer_ops)

        return qnn_arch, layers, training_data, target_unitary

    # -------------------------------------------------------------------------
    def feedforward(self,
                    samples: Iterable[Tuple[qt.Qobj, qt.Qobj]]) -> List[List[qt.Qobj]]:
        stored: List[List[qt.Qobj]] = []
        for sample, _ in samples:
            current = sample
            layerwise: List[qt.Qobj] = [sample]
            for ops in self.unitaries:
                current = self._layer_channel(ops, current)
                layerwise.append(current)
            stored.append(layerwise)
        return stored

    def _layer_channel(self,
                       ops: List[qt.Qobj],
                       input_state: qt.Qobj) -> qt.Qobj:
        # Pad input state with one extra qubit (|0>) and apply the unitary
        state = qt.tensor(input_state, qt.qeye(2))
        unitary = ops[0].copy()
        for gate in ops[1:]:
            unitary = gate * unitary
        return unitary * state

    # -------------------------------------------------------------------------
    @staticmethod
    def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
        return abs((a.dag() * b)[0, 0]) ** 2

    @staticmethod
    def fidelity_adjacency(states: Sequence[qt.Qobj],
                           threshold: float,
                           *,
                           secondary: Optional[float] = None,
                           secondary_weight: float = 0.5) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNNHybrid.state_fidelity(s_i, s_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

__all__ = ["GraphQNNHybrid", "FraudLayerParameters", "QuantumSelfAttention", "SamplerQNN"]
