import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import qiskit
from qiskit import Aer, transpile, assemble
from typing import Tuple
import torchquantum as tq
import torchquantum.functional as tqf

class QuantumGateLayer(tq.QuantumModule):
    """Simple variational layer that applies parameterised rotations and a CNOT chain."""
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
            tgt = 0 if wire == self.n_wires - 1 else wire + 1
            tqf.cnot(qdev, wires=[wire, tgt])
        return self.measure(qdev)

class QLSTM(nn.Module):
    """Classical LSTM where each gate is processed through a small quantum circuit."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.forget = QuantumGateLayer(n_qubits)
        self.input_gate = QuantumGateLayer(n_qubits)
        self.update = QuantumGateLayer(n_qubits)
        self.output = QuantumGateLayer(n_qubits)
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(self.linear_forget(combined)))
            i = torch.sigmoid(self.input_gate(self.linear_input(combined)))
            g = torch.tanh(self.update(self.linear_update(combined)))
            o = torch.sigmoid(self.output(self.linear_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        return torch.cat(outputs, dim=0), (hx, cx)

    def _init_states(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch = inputs.size(1)
        device = inputs.device
        return torch.zeros(batch, self.hidden_dim, device=device), torch.zeros(batch, self.hidden_dim, device=device)

class QuantumHybridHead(nn.Module):
    """Hybrid layer that forwards activations through a parameterised quantum circuit."""
    class QuantumCircuit:
        def __init__(self, n_qubits: int, backend, shots: int) -> None:
            self._circuit = qiskit.QuantumCircuit(n_qubits)
            all_q = list(range(n_qubits))
            self.theta = qiskit.circuit.Parameter("theta")
            self._circuit.h(all_q)
            self._circuit.barrier()
            self._circuit.ry(self.theta, all_q)
            self._circuit.measure_all()
            self.backend = backend
            self.shots = shots

        def run(self, thetas: np.ndarray) -> np.ndarray:
            compiled = transpile(self._circuit, self.backend)
            qobj = assemble(compiled, shots=self.shots, parameter_binds=[{self.theta: theta} for theta in thetas])
            job = self.backend.run(qobj)
            results = job.result().get_counts()
            expectations = []
            for counts in results.values():
                probs = np.array(list(counts.values())) / self.shots
                states = np.array([int(k, 2) for k in counts.keys()])
                expectations.append(np.sum(states * probs))
            return np.array(expectations)

    def __init__(self, n_qubits: int, backend, shots: int, shift: float) -> None:
        super().__init__()
        self.quantum_circuit = self.QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        flat = inputs.flatten()
        expectation = self.quantum_circuit.run(flat.cpu().numpy())
        return torch.tensor(expectation, device=inputs.device, dtype=inputs.dtype)

class UnifiedQLSTMNet(nn.Module):
    """
    Quantum‑enhanced sequence tagger that combines a variational LSTM
    with a parameterised quantum expectation head.
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
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        backend = qiskit.Aer.get_backend("aer_simulator")
        self.hybrid_head = QuantumHybridHead(tagset_size, backend, shots=100, shift=np.pi / 2)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        logits = self.hidden2tag(lstm_out)
        if self.hidden2tag.out_features == 1:
            batch, seq_len, _ = logits.shape
            logits_flat = logits.reshape(-1)
            probs = self.hybrid_head(logits_flat)
            probs = probs.reshape(batch, seq_len, 1)
            return torch.cat((probs, 1 - probs), dim=-1)
        else:
            return logits

__all__ = ["UnifiedQLSTMNet", "QLSTM", "QuantumHybridHead"]
