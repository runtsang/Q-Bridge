"""
HybridSelfAttentionQNN – Quantum side
Author: GPT‑QLM‑20B
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from qiskit import QuantumCircuit, QuantumRegister, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import Statevector

class HybridSelfAttentionQNN:
    """
    Quantum hybrid model that implements a QCNN‑style ansatz and
    constructs a graph of state fidelities between outputs.
    """

    def __init__(self, n_qubits: int = 4, feature_dim: int = 4):
        self.n_qubits = n_qubits
        self.feature_dim = feature_dim

        # Feature map for classical inputs
        self.feature_map = ZFeatureMap(self.feature_dim, reps=1)
        self.feature_params = self.feature_map.parameters

        # Build the QCNN ansatz
        self.ansatz = self._build_qcnn_ansatz()
        self.ansatz_params = self.ansatz.parameters

        # Full circuit: feature map + ansatz
        self.circuit = QuantumCircuit(self.feature_dim + self.n_qubits)
        # Apply feature map on first `feature_dim` qubits
        self.circuit.compose(self.feature_map, qubits=list(range(self.feature_dim)), inplace=True)
        # Apply ansatz on the remaining `n_qubits` qubits
        self.circuit.compose(self.ansatz, qubits=list(range(self.feature_dim, self.feature_dim + self.n_qubits)), inplace=True)

        # Backend
        self.backend = Aer.get_backend('statevector_simulator')

    # --------------------------------------------------------------------------- #
    # QCNN ansatz helpers
    # --------------------------------------------------------------------------- #
    def _conv_circuit(self, params: ParameterVector) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        qc.rz(-np.pi / 2, 0)
        qc.cx(0, 1)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(1, 0)
        qc.ry(params[2], 1)
        qc.cx(0, 1)
        qc.rz(np.pi / 2, 0)
        return qc

    def _conv_layer(self, num_qubits: int, prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(prefix, length=num_qubits * 3)
        for i in range(0, num_qubits, 2):
            sub = self._conv_circuit(params[i * 3 : (i + 1) * 3])
            qc.append(sub, [i, i + 1])
        return qc

    def _pool_layer(self, num_qubits: int, prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(prefix, length=num_qubits // 2 * 3)
        for i in range(0, num_qubits, 2):
            sub = self._conv_circuit(params[i // 2 * 3 : (i // 2 + 1) * 3])
            qc.append(sub, [i, i + 1])
        return qc

    def _build_qcnn_ansatz(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        # Layer 1
        qc.compose(self._conv_layer(self.n_qubits, "c1"), inplace=True)
        qc.compose(self._pool_layer(self.n_qubits, "p1"), inplace=True)
        # Layer 2
        qc.compose(self._conv_layer(self.n_qubits // 2, "c2"), inplace=True)
        qc.compose(self._pool_layer(self.n_qubits // 2, "p2"), inplace=True)
        return qc

    # --------------------------------------------------------------------------- #
    # Execution
    # --------------------------------------------------------------------------- #
    def run(self, inputs: np.ndarray) -> dict:
        """
        Execute the circuit for each input vector and return
        the resulting statevectors and a fidelity‑based graph.
        `inputs` shape: (samples, feature_dim)
        """
        states = []
        for vec in inputs:
            # Bind feature map parameters
            param_dict = {p: v for p, v in zip(self.feature_params, vec)}
            # Bind ansatz parameters (use zeros for simplicity)
            for p in self.ansatz_params:
                param_dict[p] = 0.0
            # Execute
            job = execute(self.circuit, self.backend, parameter_binds=[param_dict])
            result = job.result()
            state_vec = result.get_statevector(self.circuit)
            states.append(Statevector(state_vec))

        # Build fidelity adjacency graph
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for i, si in enumerate(states):
            for j, sj in enumerate(states[i + 1 :], start=i + 1):
                fid = si.fidelity(sj) ** 2
                if fid >= 0.8:
                    graph.add_edge(i, j, weight=1.0)
                elif fid >= 0.5:
                    graph.add_edge(i, j, weight=0.5)

        return {"states": states, "fidelity_graph": graph}

__all__ = ["HybridSelfAttentionQNN"]
