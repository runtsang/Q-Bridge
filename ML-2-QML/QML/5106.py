from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter
import networkx as nx
from dataclasses import dataclass
from typing import List, Tuple

__all__ = ["ConvFusion", "ConvFusionParameters"]

@dataclass
class ConvFusionParameters:
    kernel_size: int = 2
    shots: int = 1024
    threshold: float = 0.5
    graph_threshold: float = 0.95
    qml_variational: bool = False
    qml_variational_params: dict | None = None

class ConvFusion:
    """
    Quantum hybrid module that implements a variational quanvolution
    and provides graph utilities based on output state fidelities.
    """

    def __init__(self, params: ConvFusionParameters | None = None):
        self.params = params or ConvFusionParameters()
        self._backend = Aer.get_backend("qasm_simulator")
        self._circuit = self._build_quantum_filter()

    def _build_quantum_filter(self) -> QuantumCircuit:
        n = self.params.kernel_size ** 2
        qc = QuantumCircuit(n)
        theta = [Parameter(f"theta_{i}") for i in range(n)]
        for i in range(n):
            qc.rx(theta[i], i)
        qc.barrier()
        # Simple entangling layer
        for i in range(n - 1):
            qc.cx(i, i + 1)
        qc.barrier()
        qc.measure_all()
        return qc

    def run(self, data: np.ndarray) -> float:
        """
        Encode 2‑D data into rotation angles, execute the parameterised
        circuit, and return the average probability of measuring |1> across
        all qubits.
        """
        flat = data.reshape(-1)
        binds = []
        for val in flat:
            bind = {self._circuit.parameters[i]: np.pi if val > self.params.threshold else 0.0
                    for i in range(len(self._circuit.parameters))}
            binds.append(bind)
        job = execute(self._circuit, self._backend, shots=self.params.shots,
                      parameter_binds=binds)
        result = job.result().get_counts(self._circuit)
        total = self.params.shots * len(self._circuit.qubits)
        ones = 0
        for state, count in result.items():
            ones += sum(int(bit) for bit in state) * count
        return ones / total

    def _state_fidelity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Fidelity between two probability distributions."""
        a_norm = a / (np.linalg.norm(a) + 1e-12)
        b_norm = b / (np.linalg.norm(b) + 1e-12)
        return float(np.dot(a_norm, b_norm) ** 2)

    def _fidelity_adjacency(self, states: List[np.ndarray], threshold: float) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for i in range(len(states)):
            for j in range(i + 1, len(states)):
                fid = self._state_fidelity(states[i], states[j])
                if fid >= threshold:
                    graph.add_edge(i, j, weight=1.0)
        return graph

    def graph_regularisation(self, data: np.ndarray) -> nx.Graph:
        """
        Run the circuit on a batch of data points and build a fidelity graph
        from the resulting probability distributions.
        """
        batch = data.reshape(-1, self.params.kernel_size, self.params.kernel_size)
        states = [self._sample_distribution(sample) for sample in batch]
        return self._fidelity_adjacency(states, self.params.graph_threshold)

    def _sample_distribution(self, sample: np.ndarray) -> np.ndarray:
        flat = sample.reshape(-1)
        bind = {self._circuit.parameters[i]: np.pi if val > self.params.threshold else 0.0
                for i, val in enumerate(flat)}
        job = execute(self._circuit, self._backend, shots=self.params.shots,
                      parameter_binds=[bind])
        result = job.result().get_counts(self._circuit)
        probs = np.zeros(2 ** len(self._circuit.qubits))
        for state, count in result.items():
            idx = int(state, 2)
            probs[idx] = count / self.params.shots
        return probs
