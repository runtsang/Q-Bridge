"""Quantum implementation of GraphQNNGen111 integrating self‑attention,
convolutional layers and a variational autoencoder.
This module mirrors the classical architecture using Qiskit circuits and
quantum state evolution.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import numpy as np
import networkx as nx
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.providers.aer import QasmSimulator
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import RealAmplitudes, ZFeatureMap
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN

# --- Self‑attention quantum block -------------------------------------------

def _self_attention_circuit(n_qubits: int, rotation_params: np.ndarray,
                            entangle_params: np.ndarray) -> QuantumCircuit:
    qr = QuantumRegister(n_qubits, "q")
    cr = ClassicalRegister(n_qubits, "c")
    qc = QuantumCircuit(qr, cr)
    for i in range(n_qubits):
        qc.rx(rotation_params[3 * i], i)
        qc.ry(rotation_params[3 * i + 1], i)
        qc.rz(rotation_params[3 * i + 2], i)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
        qc.rz(entangle_params[i], i + 1)
    qc.measure(qr, cr)
    return qc

# --- QCNN convolution and pooling -----------------------------------------

def _conv_circuit(params: ParameterVector) -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    qc.cx(1, 0)
    qc.rz(np.pi / 2, 0)
    return qc

def _conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    qubits = list(range(num_qubits))
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for idx, (q1, q2) in enumerate(zip(qubits[0::2], qubits[1::2])):
        sub = _conv_circuit(params[idx * 3: idx * 3 + 3])
        qc.append(sub, [q1, q2])
    return qc

def _pool_circuit(params: ParameterVector) -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc

def _pool_layer(sources: Sequence[int], sinks: Sequence[int], param_prefix: str) -> QuantumCircuit:
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(param_prefix, length=len(sources) * 3)
    for idx, (src, snk) in enumerate(zip(sources, sinks)):
        sub = _pool_circuit(params[idx * 3: idx * 3 + 3])
        qc.append(sub, [src, snk])
    return qc

# --- Autoencoder circuit -----------------------------------------------

def _auto_encoder_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)
    # ansatz
    qc.append(RealAmplitudes(num_latent + num_trash, reps=5), range(num_latent + num_trash))
    qc.barrier()
    aux = num_latent + 2 * num_trash
    qc.h(aux)
    for i in range(num_trash):
        qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
    qc.h(aux)
    qc.measure(aux, cr[0])
    return qc

# --- Main integrated quantum architecture --------------------------------

class GraphQNNGen111Q:
    """
    Quantum counterpart of GraphQNNGen111.
    Combines a self‑attention block, QCNN convolution/pooling, and a variational
    autoencoder.  The circuit is constructed once and can be reused for
    forward evaluation or training.
    """
    def __init__(self,
                 qnn_arch: Sequence[int],
                 attention_params: np.ndarray,
                 conv_params: np.ndarray,
                 autoencoder_params: np.ndarray,
                 backend: QasmSimulator | None = None):
        self.arch = list(qnn_arch)
        self.backend = backend or Aer.get_backend("qasm_simulator")

        # Self‑attention block
        self.attention_circuit = _self_attention_circuit(
            n_qubits=len(attention_params) // 3,
            rotation_params=attention_params,
            entangle_params=attention_params[len(attention_params)//2:]
        )

        # QCNN layers
        self.conv_circuits = []
        q = 0
        for layer_size in qnn_arch[1:]:
            params = conv_params[q:q + layer_size * 3]
            self.conv_circuits.append(_conv_layer(layer_size, f"c{q}"))
            q += layer_size * 3

        # Autoencoder
        self.autoencoder_circuit = _auto_encoder_circuit(
            num_latent=int(autoencoder_params[0]),
            num_trash=int(autoencoder_params[1])
        )

        # Assemble full circuit
        self.full_circuit = QuantumCircuit()
        self.full_circuit.append(self.attention_circuit, range(self.attention_circuit.num_qubits))
        for circ in self.conv_circuits:
            self.full_circuit.append(circ, range(circ.num_qubits))
        self.full_circuit.append(self.autoencoder_circuit,
                                 range(self.autoencoder_circuit.num_qubits))

    def run(self, shots: int = 1024) -> dict:
        """Execute the full circuit and return measurement counts."""
        job = execute(self.full_circuit, self.backend, shots=shots)
        return job.result().get_counts(self.full_circuit)

    def feedforward(self, samples: Iterable[Tuple[Statevector, Statevector]]) \
            -> List[List[Statevector]]:
        """Return statevectors for each layer given a list of input states."""
        stored: List[List[Statevector]] = []
        for inp, _ in samples:
            current = inp
            layerwise = [current]
            # attention
            current = self._apply_circuit(self.attention_circuit, current)
            layerwise.append(current)
            # conv layers
            for circ in self.conv_circuits:
                current = self._apply_circuit(circ, current)
                layerwise.append(current)
            # autoencoder
            current = self._apply_circuit(self.autoencoder_circuit, current)
            layerwise.append(current)
            stored.append(layerwise)
        return stored

    @staticmethod
    def _apply_circuit(circ: QuantumCircuit, state: Statevector) -> Statevector:
        """Apply a circuit to a statevector and return the new state."""
        new_state = state.evolve(circ)
        return new_state

    @staticmethod
    def state_fidelity(a: Statevector, b: Statevector) -> float:
        return abs((a.dag() @ b)[0, 0]) ** 2

    @staticmethod
    def fidelity_adjacency(states: Sequence[Statevector], threshold: float,
                           *, secondary: float | None = None,
                           secondary_weight: float = 0.5) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNNGen111Q.state_fidelity(a, b)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    @staticmethod
    def random_network(qnn_arch: Sequence[int], samples: int):
        """Generate random parameters for each sub‑module."""
        # Random attention params
        att_params = np.random.randn(len(qnn_arch) * 3)
        # Random convolution params
        conv_params = np.random.randn(sum(qnn_arch[1:]) * 3)
        # Random autoencoder params (latent, trash)
        auto_params = np.random.randint(1, 5, size=2)
        # Training data: for quantum we use random statevectors
        training_data = []
        for _ in range(samples):
            state = Statevector.random(2 ** len(att_params) // 3)
            target = state  # identity target for placeholder
            training_data.append((state, target))
        return list(qnn_arch), att_params, conv_params, auto_params, training_data

__all__ = [
    "GraphQNNGen111Q",
    "random_network",
]
