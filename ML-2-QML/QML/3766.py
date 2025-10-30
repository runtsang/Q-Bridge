"""Quantum‑enhanced LSTM with single‑qubit parameterised gates inspired by FCL.

Each LSTM gate is realised by a miniature quantum circuit that applies a
Hadamard, a parametrised Ry rotation, and measures the qubit in the Z basis.
The expectation value of the measurement is then passed through the usual
sigmoid or tanh non‑linearities to produce gate activations.
The implementation uses Qiskit Aer for simulation and retains the
original QLSTM API for seamless substitution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from typing import Tuple, Optional


class QuantumFullyConnectedGate:
    """Parameterised single‑qubit gate that returns the probability of measuring |1>."""
    def __init__(self, n_qubits: int = 1, backend=None, shots: int = 1024):
        self.n_qubits = n_qubits
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots
        # Base circuit template (no parameter yet)
        self.base_circuit = qiskit.QuantumCircuit(n_qubits)
        self.base_circuit.h(range(n_qubits))
        self.base_circuit.barrier()
        self.theta = qiskit.circuit.Parameter("theta")
        for i in range(n_qubits):
            self.base_circuit.ry(self.theta, i)
        self.base_circuit.measure_all()

    def _expectation(self, theta: float) -> float:
        """Run the circuit for a single theta and return the |1> probability."""
        circuit = self.base_circuit.copy()
        bound_circuit = circuit.bind_parameters({self.theta: theta})
        job = qiskit.execute(bound_circuit, self.backend, shots=self.shots)
        result = job.result()
        counts = result.get_counts(bound_circuit)
        total = sum(counts.values())
        exp = sum(int(bits, 2) * cnt for bits, cnt in counts.items()) / total
        return exp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute expectations for a batch of scalars."""
        batch = x.squeeze().tolist()
        exps = [self._expectation(float(v)) for v in batch]
        return torch.tensor(exps, dtype=torch.float32, device=x.device).unsqueeze(-1)


class QLSTM(nn.Module):
    """Quantum‑enhanced LSTM cell mirroring the classical API."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 1, shots: int = 1024, backend=None) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        gate_input = input_dim + hidden_dim
        self.linear_forget = nn.Linear(gate_input, n_qubits)
        self.linear_input = nn.Linear(gate_input, n_qubits)
        self.linear_update = nn.Linear(gate_input, n_qubits)
        self.linear_output = nn.Linear(gate_input, n_qubits)

        self.forget_gate = QuantumFullyConnectedGate(n_qubits, backend, shots)
        self.input_gate = QuantumFullyConnectedGate(n_qubits, backend, shots)
        self.update_gate = QuantumFullyConnectedGate(n_qubits, backend, shots)
        self.output_gate = QuantumFullyConnectedGate(n_qubits, backend, shots)

    def _init_states(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

    def forward(self, inputs: torch.Tensor, states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size = inputs.size(1)
        device = inputs.device
        hx, cx = self._init_states(batch_size, device) if states is None else states

        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_gate(self.linear_forget(combined)))
            i = torch.sigmoid(self.input_gate(self.linear_input(combined)))
            g = torch.tanh(self.update_gate(self.linear_update(combined)))
            o = torch.sigmoid(self.output_gate(self.linear_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        return torch.cat(outputs, dim=0), (hx, cx)


class LSTMTagger(nn.Module):
    """Sequence tagging model that uses the quantum LSTM cell."""
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int, tagset_size: int, n_qubits: int = 1, shots: int = 1024, backend=None) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits, shots=shots, backend=backend)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.unsqueeze(1))
        tag_logits = self.hidden2tag(lstm_out.squeeze(1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["QLSTM", "LSTMTagger"]
