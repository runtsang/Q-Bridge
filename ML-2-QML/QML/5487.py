"""Quantum implementation of the hybrid module.

The structure mirrors the classical implementation but replaces each sub‑module with a parameter‑driven quantum circuit."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
import torchquantum as tq
import torchquantum.functional as tqf

class QuantumFCL(nn.Module):
    """Parameterized quantum circuit acting as a fully‑connected layer."""
    def __init__(self, n_qubits: int = 1, shots: int = 1024):
        super().__init__()
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = Aer.get_backend("qasm_simulator")
        self.circuit = QuantumCircuit(n_qubits)
        self.theta = ParameterVector("theta", n_qubits)
        for q, t in zip(range(n_qubits), self.theta):
            self.circuit.h(q)
            self.circuit.ry(t, q)
        self.circuit.measure_all()

    def forward(self, thetas: torch.Tensor) -> torch.Tensor:
        param_binds = [{self.theta[i]: float(t) for i in range(self.n_qubits)}]
        job = execute(self.circuit, self.backend, shots=self.shots, parameter_binds=param_binds)
        result = job.result()
        counts = result.get_counts()
        probs = np.array([counts.get(bin(i)[2:].zfill(self.n_qubits), 0) for i in range(2**self.n_qubits)]) / self.shots
        expectation = np.sum(probs * np.arange(2**self.n_qubits))
        return torch.tensor([expectation], dtype=torch.float32)

class QuantumEstimatorQNN(nn.Module):
    """Quantum estimator that mimics the feed‑forward EstimatorQNN."""
    def __init__(self, n_qubits: int = 2):
        super().__init__()
        self.n_qubits = n_qubits
        self.circuit = QuantumCircuit(n_qubits)
        self.input_params = ParameterVector("x", n_qubits)
        self.weight_params = ParameterVector("w", n_qubits)
        for q in range(n_qubits):
            self.circuit.h(q)
            self.circuit.ry(self.input_params[q], q)
            self.circuit.rz(self.weight_params[q], q)
        self.circuit.measure_all()
        self.backend = Aer.get_backend("qasm_simulator")
        self.shots = 1024

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        param_bind = {self.input_params[i]: float(inputs[i]) for i in range(self.n_qubits)}
        param_bind.update({self.weight_params[i]: float(inputs[i]) for i in range(self.n_qubits)})
        job = execute(self.circuit, self.backend, shots=self.shots, parameter_binds=[param_bind])
        result = job.result()
        counts = result.get_counts()
        probs = np.array([counts.get(bin(i)[2:].zfill(self.n_qubits), 0) for i in range(2**self.n_qubits)]) / self.shots
        expectation = np.sum(probs * np.arange(2**self.n_qubits))
        return torch.tensor([expectation], dtype=torch.float32)

def build_classifier_circuit_qml(num_qubits: int, depth: int):
    """Quantum classifier ansatz mirroring the classical incremental circuit."""
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)
    circuit = QuantumCircuit(num_qubits)
    for q, p in zip(range(num_qubits), encoding):
        circuit.rx(p, q)
    idx = 0
    for _ in range(depth):
        for q in range(num_qubits):
            circuit.ry(weights[idx], q)
            idx += 1
        for q in range(num_qubits - 1):
            circuit.cz(q, q + 1)
    circuit.measure_all()
    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return circuit, list(encoding), list(weights), observables

class QuantumQLSTM(nn.Module):
    """Quantum‑enhanced LSTM cell implemented with torchquantum."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
            self.encoder(qdev, x)
            for gate, wire in zip(self.params, range(self.n_wires)):
                gate(qdev, wires=wire)
            for wire in range(self.n_wires - 1):
                tqf.cnot(qdev, wires=[wire, wire + 1])
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.forget = self.QLayer(n_qubits)
        self.input = self.QLayer(n_qubits)
        self.update = self.QLayer(n_qubits)
        self.output = self.QLayer(n_qubits)
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(self, inputs: torch.Tensor, states: tuple | None = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(self.linear_forget(combined)))
            i = torch.sigmoid(self.input(self.linear_input(combined)))
            g = torch.tanh(self.update(self.linear_update(combined)))
            o = torch.sigmoid(self.output(self.linear_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(self, inputs: torch.Tensor, states: tuple | None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

class QuantumLSTMTagger(nn.Module):
    """Tagger that uses the quantum LSTM."""
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int, tagset_size: int, n_qubits: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QuantumQLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

class HybridFCLClassifierQLSTM(nn.Module):
    """Quantum hybrid model mirroring the classical counterpart."""
    def __init__(self, config: dict):
        super().__init__()
        self.fcl = QuantumFCL(n_qubits=config.get("fcl_qubits", 1))
        self.estimator = QuantumEstimatorQNN(n_qubits=config.get("est_qubits", 2))
        self.circuit, self.enc, self.w_sizes, self.obs = build_classifier_circuit_qml(
            num_qubits=config.get("clf_qubits", 1),
            depth=config.get("clf_depth", 2)
        )
        self.circuit_backend = Aer.get_backend("qasm_simulator")
        self.tagger = QuantumLSTMTagger(embedding_dim=config.get("emb_dim", 8),
                                        hidden_dim=config.get("lstm_hidden", 16),
                                        vocab_size=config.get("vocab_size", 1000),
                                        tagset_size=config.get("tagset_size", 10),
                                        n_qubits=config.get("n_qubits", 4))

    def forward(self, x: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
        """Quantum forward pass."""
        fcl_out = self.fcl(x)
        est_out = self.estimator(fcl_out)
        param_binds = [{self.enc[0]: float(est_out[0])}]
        job = execute(self.circuit, self.circuit_backend, shots=1024, parameter_binds=param_binds)
        result = job.result()
        counts = result.get_counts()
        probs = np.array([counts.get(bin(i)[2:].zfill(self.circuit.num_qubits), 0) for i in range(2**self.circuit.num_qubits)]) / 1024
        clf_out = torch.tensor([np.sum(probs * np.arange(2**self.circuit.num_qubits))], dtype=torch.float32)
        tag_logits = self.tagger(seq)
        return torch.cat([clf_out, tag_logits], dim=-1)

__all__ = ["HybridFCLClassifierQLSTM"]
