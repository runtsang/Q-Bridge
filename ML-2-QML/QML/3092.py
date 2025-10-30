from qiskit import QuantumCircuit, ParameterVector
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QuantumGateSampler(nn.Module):
    """Quantum sampler that produces a probability vector for LSTM gates."""
    def __init__(self, gate_dim: int) -> None:
        super().__init__()
        self.gate_dim = gate_dim

        # Parameter vectors for inputs and trainable weights
        self.inputs = ParameterVector('x', gate_dim)
        self.weights = ParameterVector('w', gate_dim)

        # Build a simple parameterized circuit
        qc = QuantumCircuit(gate_dim)
        for i in range(gate_dim):
            qc.ry(self.inputs[i], i)
            qc.ry(self.weights[i], i)
            if i < gate_dim - 1:
                qc.cx(i, i + 1)

        # Sampler primitive
        sampler = Sampler()
        self.sampler_qnn = SamplerQNN(
            circuit=qc,
            input_params=self.inputs,
            weight_params=self.weights,
            sampler=sampler
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, gate_dim)
        # Convert to numpy for Qiskit
        x_np = x.detach().cpu().numpy()
        # Execute sampler: returns probabilities for each basis state
        probs = self.sampler_qnn.forward(x_np)  # shape (batch, gate_dim)
        return torch.tensor(probs, device=x.device, dtype=x.dtype)


class HybridQLSTM(nn.Module):
    """Hybrid LSTM with fully quantum gates."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.input_dim = input_dim

        # Linear projections before quantum sampling
        self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # Quantum gate samplers
        self.forget_sampler = QuantumGateSampler(hidden_dim)
        self.input_sampler = QuantumGateSampler(hidden_dim)
        self.update_sampler = QuantumGateSampler(hidden_dim)
        self.output_sampler = QuantumGateSampler(hidden_dim)

    def forward(self, inputs: torch.Tensor,
                states: tuple | None = None) -> tuple:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)

            f_logits = self.forget_linear(combined)
            i_logits = self.input_linear(combined)
            g_logits = self.update_linear(combined)
            o_logits = self.output_linear(combined)

            f = torch.sigmoid(self.forget_sampler(f_logits))
            i = torch.sigmoid(self.input_sampler(i_logits))
            g = torch.tanh(self.update_sampler(g_logits))
            o = torch.sigmoid(self.output_sampler(o_logits))

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(self, inputs: torch.Tensor,
                     states: tuple | None) -> tuple:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx


class HybridLSTMTagger(nn.Module):
    """Sequence tagging model using the quantum HybridQLSTM."""
    def __init__(self, embedding_dim: int, hidden_dim: int,
                 vocab_size: int, tagset_size: int, n_qubits: int) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = HybridQLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["HybridQLSTM", "HybridLSTMTagger"]
