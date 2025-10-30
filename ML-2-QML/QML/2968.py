import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from qiskit import QuantumCircuit, Aer, transpile
from qiskit.quantum_info import Statevector
from qiskit.circuit import Parameter
from qiskit.quantum_info.operators import Pauli
from typing import Tuple, Iterable, Callable, List

class HybridQLSTM(nn.Module):
    """Quantum‑enhanced LSTM using Qiskit circuits for the gates."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.use_quantum = n_qubits > 0

        self.linear_forget = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.linear_input = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.linear_update = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.linear_output = nn.Linear(input_dim + hidden_dim, hidden_dim)

        if self.use_quantum:
            self._gate_template = self._build_gate_template(n_qubits)
            self.simulator = Aer.get_backend('statevector_simulator')

    def _build_gate_template(self, n_wires: int) -> QuantumCircuit:
        angles = [Parameter(f'θ_{i}') for i in range(n_wires)]
        qc = QuantumCircuit(n_wires)
        for i, a in enumerate(angles):
            qc.rx(a, i)
        for i in range(n_wires):
            target = 0 if i == n_wires - 1 else i + 1
            qc.cx(i, target)
        return qc

    def _apply_quantum_gate(self, angles: torch.Tensor) -> torch.Tensor:
        batch = angles.shape[0]
        results = []
        for i in range(batch):
            param_dict = {f'θ_{j}': float(angles[i, j].item()) for j in range(self.n_qubits)}
            qc = self._gate_template.bind_parameters(param_dict)
            qc = transpile(qc, self.simulator)
            state = Statevector.from_instruction(qc)
            exp = []
            for q in range(self.n_qubits):
                pauli = Pauli('Z' * q + 'I' * (self.n_qubits - q - 1))
                exp.append(state.expectation_value(pauli).real)
            results.append(exp)
        return torch.tensor(results, device=angles.device, dtype=angles.dtype)

    def _init_states(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f_lin = self.linear_forget(combined)
            i_lin = self.linear_input(combined)
            g_lin = self.linear_update(combined)
            o_lin = self.linear_output(combined)

            if self.use_quantum:
                f = torch.sigmoid(self._apply_quantum_gate(f_lin))
                i = torch.sigmoid(self._apply_quantum_gate(i_lin))
                g = torch.tanh(self._apply_quantum_gate(g_lin))
                o = torch.sigmoid(self._apply_quantum_gate(o_lin))
            else:
                f = torch.sigmoid(f_lin)
                i = torch.sigmoid(i_lin)
                g = torch.tanh(g_lin)
                o = torch.sigmoid(o_lin)

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        return torch.cat(outputs, dim=0), (hx, cx)

    def evaluate(
        self,
        inputs: torch.Tensor,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor]],
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        self.eval()
        with torch.no_grad():
            outputs, _ = self.forward(inputs)
            raw = [obs(outputs) for obs in observables]
            scalars = [torch.mean(r, dim=0).cpu().numpy().tolist() for r in raw]
            if shots is not None:
                rng = np.random.default_rng(seed)
                noisy = [[rng.normal(mean, max(1e-6, 1 / shots)) for mean in row] for row in scalars]
                return noisy
            return scalars

class LSTMTagger(nn.Module):
    """Sequence tagging model that uses :class:`HybridQLSTM`."""
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = HybridQLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["HybridQLSTM", "LSTMTagger"]
