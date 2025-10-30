from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import itertools
from typing import Iterable

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector, Parameter
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.primitives import StatevectorSampler as Sampler, StatevectorEstimator as Estimator

import torchquantum as tq
import torchquantum.functional as tqf

class HybridSamplerEstimatorQLSTMGraphNet(nn.Module):
    """
    Quantum‑centric hybrid network that mirrors the classical
    HybridSamplerEstimatorQLSTMGraphNet.  Each component is implemented
    with parameterised quantum circuits or quantum modules.
    """

    def __init__(self,
                 input_dim: int = 2,
                 hidden_dim: int = 4,
                 n_qubits: int = 4,
                 graph_threshold: float = 0.8) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.graph_threshold = graph_threshold

        # Quantum sampler circuit
        inputs = ParameterVector("input", input_dim)
        weights = ParameterVector("weight", 4)
        qc = QuantumCircuit(input_dim)
        for i in range(input_dim):
            qc.ry(inputs[i], i)
        qc.cx(0, 1)
        for w in weights:
            qc.ry(w, 0)
        self.sampler_circuit = qc
        self.sampler_weights = weights

        # Quantum estimator circuit
        inp = Parameter("input1")
        wgt = Parameter("weight1")
        qc_est = QuantumCircuit(1)
        qc_est.h(0)
        qc_est.ry(inp, 0)
        qc_est.rx(wgt, 0)
        self.estimator_circuit = qc_est
        self.estimator_weight = wgt

        # Quantum LSTM cell
        class QLayer(tq.QuantumModule):
            def __init__(self, n_wires: int) -> None:
                super().__init__()
                self.encoder = tq.GeneralEncoder(
                    [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
                )
                self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
                self.measure = tq.MeasureAll(tq.PauliZ)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                qdev = tq.QuantumDevice(n_wires=self.encoder.n_wires, bsz=x.shape[0], device=x.device)
                self.encoder(qdev, x)
                for gate in self.params:
                    gate(qdev)
                for i in range(self.encoder.n_wires - 1):
                    tqf.cnot(qdev, wires=[i, i + 1])
                tqf.cnot(qdev, wires=[self.encoder.n_wires - 1, 0])
                return self.measure(qdev)

        self.forget = QLayer(n_qubits)
        self.input_gate = QLayer(n_qubits)
        self.update = QLayer(n_qubits)
        self.output_gate = QLayer(n_qubits)

        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def _quantum_lstm(self, inputs: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        hx = torch.zeros(inputs.size(0), self.hidden_dim, device=inputs.device)
        cx = torch.zeros(inputs.size(0), self.hidden_dim, device=inputs.device)
        outputs = []
        for x in inputs.unbind(dim=1):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(self.linear_forget(combined)))
            i = torch.sigmoid(self.input_gate(self.linear_input(combined)))
            g = torch.tanh(self.update(self.linear_update(combined)))
            o = torch.sigmoid(self.output_gate(self.linear_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(1))
        return torch.cat(outputs, dim=1), (hx, cx)

    def _simulate_sampler(self, params: torch.Tensor) -> torch.Tensor:
        """Return a 2‑element probability vector from the sampler circuit."""
        qc = self.sampler_circuit.copy()
        for i, val in enumerate(params):
            qc.ry(val.item(), i)
        for w in self.sampler_weights:
            qc.ry(w.item(), 0)
        sv = Statevector.from_instruction(qc)
        probs = sv.probabilities()
        prob0 = probs[0] + probs[2]   # |00> and |10>
        prob1 = probs[1] + probs[3]   # |01> and |11>
        return torch.tensor([prob0, prob1], dtype=torch.float32)

    def _simulate_estimator(self, params: torch.Tensor) -> torch.Tensor:
        """Return a 1‑element expectation value from the estimator circuit."""
        qc = self.estimator_circuit.copy()
        qc.ry(params[0].item(), 0)
        qc.rx(self.estimator_weight.item(), 0)
        sv = Statevector.from_instruction(qc)
        exp_y = sv.expectation_value(SparsePauliOp.from_list([("Y", 1)]))
        return torch.tensor([exp_y], dtype=torch.float32)

    def _state_fidelity(self, a: Statevector, b: Statevector) -> float:
        return abs((a.data.conj().T @ b.data)[0]) ** 2

    def _fidelity_adjacency(self,
                            states: Iterable[Statevector],
                            threshold: float,
                            *,
                            secondary: float | None = None,
                            secondary_weight: float = 0.5) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
            fid = self._state_fidelity(s_i, s_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    def forward(self, inputs: torch.Tensor) -> dict:
        """
        Parameters
        ----------
        inputs : torch.Tensor
            Sequence of shape (batch, seq_len, input_dim).

        Returns
        -------
        dict
            Dictionary containing sampler logits, sampler probabilities,
            estimator outputs, hidden states and a graph of state fidelities.
        """
        lstm_out, _ = self._quantum_lstm(inputs)

        sampler_logits = []
        sampler_probs = []
        estimator_out = []

        for h in lstm_out.unbind(dim=1):
            # h shape: (batch, hidden_dim)
            params = h[:, :self.input_dim]
            samp_batch = torch.stack([self._simulate_sampler(p) for p in params], dim=0)
            sampler_logits.append(samp_batch)
            sampler_probs.append(F.softmax(samp_batch, dim=-1))
            est_batch = torch.stack([self._simulate_estimator(p) for p in params], dim=0)
            estimator_out.append(est_batch)

        sampler_logits = torch.stack(sampler_logits, dim=1)  # (batch, seq_len, 2)
        sampler_probs = torch.stack(sampler_probs, dim=1)    # (batch, seq_len, 2)
        estimator_out = torch.stack(estimator_out, dim=1)    # (batch, seq_len, 1)

        # Fidelity graph of hidden states (classical proxy)
        hidden_states = [Statevector.from_label('0'*self.n_qubits) for _ in range(lstm_out.size(1))]
        graph = self._fidelity_adjacency(hidden_states, self.graph_threshold)

        return {
            "sampler_logits": sampler_logits,
            "sampler_probs": sampler_probs,
            "estimator_out": estimator_out,
            "hidden_states": lstm_out,
            "fidelity_graph": graph
        }

__all__ = ["HybridSamplerEstimatorQLSTMGraphNet"]
