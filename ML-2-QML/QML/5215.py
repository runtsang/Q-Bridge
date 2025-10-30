from __future__ import annotations

import qiskit
import numpy as np
import networkx as nx
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.circuit.random import random_circuit
from qiskit import Aer, execute
import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
from typing import Sequence, Iterable, Callable, List

# Quantum estimator for expectation values
class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrized circuit."""
    def __init__(self, circuit: QuantumCircuit):
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

class GenConv096:
    """
    Quantum hybrid convolution filter (quanvolution) with optional quantum LSTM gating,
    fidelity‑based graph adjacency, and FastBaseEstimator support.
    """
    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 127.0,
        shots: int = 100,
        use_lstm: bool = False,
        use_graph: bool = False,
        graph_threshold: float = 0.8,
    ) -> None:
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.shots = shots
        self.backend = Aer.get_backend("qasm_simulator")
        self.graph_threshold = graph_threshold

        self.circuit = self._build_circuit()
        self.use_lstm = use_lstm
        if use_lstm:
            self.lstm_gate = self._build_lstm_gate()

        self.use_graph = use_graph

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i, t in enumerate(self.theta):
            qc.rx(t, i)
        qc.barrier()
        qc += random_circuit(self.n_qubits, 2)
        qc.measure_all()
        return qc

    def _build_lstm_gate(self) -> tq.QuantumModule:
        class QLayer(tq.QuantumModule):
            def __init__(self, n_wires: int):
                super().__init__()
                self.n_wires = n_wires
                self.encoder = tq.GeneralEncoder(
                    [
                        {"input_idx": [0], "func": "rx", "wires": [0]},
                        {"input_idx": [1], "func": "rx", "wires": [1]},
                        {"input_idx": [2], "func": "rx", "wires": [2]},
                        {"input_idx": [3], "func": "rx", "wires": [3]},
                    ]
                )
                self.params = nn.ModuleList(
                    [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
                )
                self.measure = tq.MeasureAll(tq.PauliZ)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                qdev = tq.QuantumDevice(
                    n_wires=self.n_wires, bsz=x.shape[0], device=x.device
                )
                self.encoder(qdev, x)
                for w, gate in enumerate(self.params):
                    gate(qdev, wires=w)
                for w in range(self.n_wires):
                    tgt = 0 if w == self.n_wires - 1 else w + 1
                    tqf.cnot(qdev, wires=[w, tgt])
                return self.measure(qdev)
        return QLayer(self.n_qubits)

    def run(self, data: np.ndarray) -> float:
        """
        Execute the quanvolution circuit on a 2D patch and return the average |1> probability.
        """
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)

        job = execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result().get_counts(self.circuit)

        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val

        prob = counts / (self.shots * self.n_qubits)

        if self.use_lstm:
            prob_tensor = torch.tensor([prob], dtype=torch.float32).unsqueeze(0)
            lstm_out = self.lstm_gate(prob_tensor)
            prob = lstm_out.squeeze().item()

        if self.use_graph:
            probs = torch.tensor([prob], dtype=torch.float32).unsqueeze(0)
            graph = nx.Graph()
            graph.add_nodes_from(range(probs.size(0)))
            for i in range(probs.size(0)):
                for j in range(i + 1, probs.size(0)):
                    fid = torch.dot(probs[i], probs[j]) / (
                        torch.norm(probs[i]) * torch.norm(probs[j]) + 1e-12
                    )
                    if fid >= self.graph_threshold:
                        graph.add_edge(i, j, weight=1.0)
            agg = torch.zeros_like(probs)
            for comp in nx.connected_components(graph):
                idx = list(comp)
                agg[idx] = probs[idx].mean(dim=0, keepdim=True)
            prob = agg.squeeze().item()

        return prob

    def evaluate_expectations(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Delegate to FastBaseEstimator for expectation evaluation."""
        estimator = FastBaseEstimator(self.circuit)
        return estimator.evaluate(observables, parameter_sets)
