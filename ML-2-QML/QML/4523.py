"""Quantum‑enhanced fraud detection model that integrates a quantum LSTM, a quantum graph layer, and a quantum expectation head."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
from qiskit import Aer, QuantumCircuit, assemble, transpile
from qiskit.circuit import Parameter


class QuantumGraphLayer(tq.QuantumModule):
    """Quantum graph layer that encodes an adjacency matrix through controlled rotations."""
    def __init__(self, num_nodes: int, adjacency: np.ndarray):
        super().__init__()
        self.num_nodes = num_nodes
        self.adjacency = adjacency
        self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(num_nodes)])
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(num_nodes)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qdev = tq.QuantumDevice(n_wires=self.num_nodes, bsz=x.shape[0], device=x.device)
        self.encoder(qdev, x)
        for i, gate in enumerate(self.params):
            gate(qdev, wires=[i])
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if self.adjacency[i, j] > 0:
                    tqf.cnot(qdev, wires=[i, j])
        return tq.MeasureAll(tq.PauliZ)(qdev)


class QuantumLSTMCell(tq.QuantumModule):
    """Quantum LSTM cell with parameterized gates."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.forget_gate = self._make_gate()
        self.input_gate = self._make_gate()
        self.update_gate = self._make_gate()
        self.output_gate = self._make_gate()
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def _make_gate(self) -> tq.QuantumModule:
        n_wires = self.n_qubits
        return tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
        )

    def forward(self, x: torch.Tensor, hx: torch.Tensor, cx: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([x, hx], dim=1)
        f = torch.sigmoid(self.forget_gate(self.linear_forget(combined)))
        i = torch.sigmoid(self.input_gate(self.linear_input(combined)))
        g = torch.tanh(self.update_gate(self.linear_update(combined)))
        o = torch.sigmoid(self.output_gate(self.linear_output(combined)))
        cx = f * cx + i * g
        hx = o * torch.tanh(cx)
        return hx, cx


class QuantumHybridHead(nn.Module):
    """Quantum expectation head that maps a classical vector to a probability."""
    def __init__(self, in_features: int, n_qubits: int, backend=None, shots: int = 200):
        super().__init__()
        self.n_qubits = n_qubits
        self.backend = backend or Aer.get_backend("aer_simulator")
        self.shots = shots
        self.theta = Parameter("theta")
        self.circuit = QuantumCircuit(n_qubits)
        self.circuit.h(range(n_qubits))
        self.circuit.ry(self.theta, range(n_qubits))
        self.circuit.measure_all()
        self.circuit = transpile(self.circuit, self.backend)

    def _run(self, thetas: np.ndarray) -> np.ndarray:
        qobj = assemble(self.circuit, shots=self.shots,
                        parameter_binds=[{self.theta: t} for t in thetas])
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array(list(count_dict.keys())).astype(float)
            probs = counts / self.shots
            return np.sum(states * probs)
        if isinstance(result, list):
            return np.array([expectation(r) for r in result])
        return np.array([expectation(result)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Map first feature to rotation angles
        angles = x[:, 0].cpu().numpy()
        exp_vals = self._run(angles)
        return torch.sigmoid(torch.tensor(exp_vals, device=x.device, dtype=torch.float32))


class FraudDetectionHybrid(nn.Module):
    """Quantum‑enhanced fraud detection model."""
    def __init__(self,
                 input_dim: int,
                 lstm_hidden: int,
                 graph_nodes: int,
                 graph_adj: np.ndarray,
                 lstm_qubits: int,
                 graph_qubits: int,
                 head_qubits: int,
                 backend=None,
                 shots: int = 200):
        super().__init__()
        self.lstm_cell = QuantumLSTMCell(input_dim, lstm_hidden, lstm_qubits)
        self.graph_layer = QuantumGraphLayer(graph_nodes, graph_adj)
        self.head = QuantumHybridHead(lstm_hidden + graph_nodes, head_qubits, backend, shots)

    def forward(self, seq: torch.Tensor, graph_features: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        seq : torch.Tensor
            Transaction sequence of shape (batch, seq_len, input_dim).
        graph_features : torch.Tensor
            Node features of graph shape (batch, num_nodes, feature_dim).
        Returns
        -------
        torch.Tensor
            Fraud probability of shape (batch,).
        """
        batch_size, seq_len, _ = seq.size()
        hx = torch.zeros(batch_size, self.lstm_cell.hidden_dim, device=seq.device)
        cx = torch.zeros(batch_size, self.lstm_cell.hidden_dim, device=seq.device)
        for t in range(seq_len):
            hx, cx = self.lstm_cell(seq[:, t, :], hx, cx)
        graph_emb = self.graph_layer(graph_features.view(-1, graph_features.size(-1)))
        graph_emb = graph_emb.mean(dim=0).unsqueeze(0).expand(batch_size, -1)
        combined = torch.cat([hx, graph_emb], dim=1)
        probs = self.head(combined)
        return probs


__all__ = ["FraudDetectionHybrid"]
