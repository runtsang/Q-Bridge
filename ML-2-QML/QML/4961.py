"""
Hybrid LSTM tagger – quantum implementation.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit.primitives import StatevectorSampler, StatevectorEstimator


# --------------------------------------------------------------------------- #
#  Quantum self‑attention block (Qiskit based)
# --------------------------------------------------------------------------- #
class QuantumSelfAttention:
    """
    Implements a toy self‑attention style quantum circuit.
    Uses rotation gates for query/key/value generation and a
    controlled‑rotation entangling layer.
    """
    def __init__(self, n_qubits: int = 4) -> None:
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)
        # Encode rotation parameters
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        # Entangling layer
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(self, backend, rotation_params: np.ndarray, entangle_params: np.ndarray, shots: int = 1024) -> dict:
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = backend.run(circuit, shots=shots)
        return job.result().get_counts(circuit)


# --------------------------------------------------------------------------- #
#  Quantum sampler (Qiskit Machine Learning)
# --------------------------------------------------------------------------- #
def QuantumSamplerQNN() -> SamplerQNN:
    inputs = ParameterVector("input", 2)
    weights = ParameterVector("weight", 4)
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
    return SamplerQNN(circuit=qc, input_params=inputs, weight_params=weights, sampler=sampler)


# --------------------------------------------------------------------------- #
#  Quantum estimator (Qiskit Machine Learning)
# --------------------------------------------------------------------------- #
def QuantumEstimatorQNN() -> EstimatorQNN:
    inp = ParameterVector("input", 1)
    wgt = ParameterVector("weight", 1)
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.ry(inp[0], 0)
    qc.rx(wgt[0], 0)
    observable = tq.SparsePauliOp.from_list([("Y", 1)])
    estimator = StatevectorEstimator()
    return EstimatorQNN(circuit=qc, observables=observable,
                        input_params=[inp[0]], weight_params=[wgt[0]], estimator=estimator)


# --------------------------------------------------------------------------- #
#  Quantum LSTM cell (TorchQuantum based)
# --------------------------------------------------------------------------- #
class QuantumQLSTM(nn.Module):
    """
    LSTM cell where each gate is a small variational circuit.
    """
    class QGate(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
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
            self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            for wire in range(self.n_wires):
                if wire == self.n_wires - 1:
                    tqf.cnot(qdev, wires=[wire, 0])
                else:
                    tqf.cnot(qdev, wires=[wire, wire + 1])
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.forget = self.QGate(n_qubits)
        self.input_gate = self.QGate(n_qubits)
        self.update = self.QGate(n_qubits)
        self.output = self.QGate(n_qubits)
        self.lin_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.lin_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.lin_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.lin_output = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.dropout = nn.Dropout(0.1)

    def forward(self, inputs: torch.Tensor,
                states: tuple[torch.Tensor, torch.Tensor] | None = None) \
            -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(self.lin_forget(combined)))
            i = torch.sigmoid(self.input_gate(self.lin_input(combined)))
            g = torch.tanh(self.update(self.lin_update(combined)))
            o = torch.sigmoid(self.output(self.lin_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(self.dropout(hx.unsqueeze(0)))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(self, inputs: torch.Tensor,
                     states: tuple[torch.Tensor, torch.Tensor] | None) \
            -> tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))


# --------------------------------------------------------------------------- #
#  Hybrid quantum tagger
# --------------------------------------------------------------------------- #
class HybridLSTMTagger(nn.Module):
    """
    Quantum‑enhanced sequence‑tagger that fuses:
      * quantum self‑attention
      * quantum sampler network
      * quantum estimator (regression)
      * variational LSTM cell
      * classical output head
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 4,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.attention = QuantumSelfAttention(n_qubits)
        self.sampler = QuantumSamplerQNN()
        self.estimator = QuantumEstimatorQNN()
        self.lstm = QuantumQLSTM(embedding_dim, hidden_dim, n_qubits)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Returns:
            logits: (seq_len, tagset_size) – log‑softmax of tag scores
            regression: (batch, 1) – expectation value from the estimator
        """
        embeds = self.embedding(sentence)                # (seq_len, batch, embed_dim)

        # Quantum self‑attention – here we simply fake the parameters
        # for illustration; in practice they would be trainable.
        rot = np.random.randn(12)
        ent = np.random.randn(3)
        backend = qiskit.Aer.get_backend("qasm_simulator")
        attn_counts = self.attention.run(backend, rot, ent)
        # Convert counts to a tensor (placeholder)
        attn_tensor = torch.tensor(list(attn_counts.values()), dtype=torch.float32).unsqueeze(1)

        # Combine embeddings with attention output
        attn_emb = torch.cat([embeds, attn_tensor], dim=-1)

        # Sample a weight vector from the quantum sampler
        sample_counts = self.sampler.run(
            backend, np.random.randn(2), np.random.randn(4), shots=1024)
        weight_vec = torch.tensor(list(sample_counts.values()), dtype=torch.float32).unsqueeze(0)
        weighted = attn_emb * weight_vec

        # Variational LSTM
        lstm_out, _ = self.lstm(weighted)

        # Classification logits
        logits = F.log_softmax(self.hidden2tag(lstm_out), dim=-1)

        # Regression head via estimator
        reg_counts = self.estimator.run(backend, np.random.randn(1), np.random.randn(1), shots=1024)
        regression = torch.tensor(list(reg_counts.values()), dtype=torch.float32).unsqueeze(-1)

        return {"logits": logits, "regression": regression}


__all__ = ["HybridLSTMTagger", "QuantumQLSTM", "QuantumSelfAttention",
           "QuantumSamplerQNN", "QuantumEstimatorQNN"]
