"""Hybrid self‑attention LSTM classifier with quantum components.

Each classical block is replaced by a parameterised quantum circuit or a quantum‑inspired module:
* QuantumSelfAttention – a simple circuit that produces a scalar weight per token.
* QuantumQLSTM – gates realised by small quantum circuits.
* QuantumFCL – a single‑qubit circuit that returns a tanh‑activated expectation.
* QuantumHybrid – a hybrid head that forwards activations through a parameterised circuit.

The overall interface mirrors the classical version, accepting a token sequence and emitting
a probability distribution over two classes.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import assemble, transpile

# --------------------------------------------------------------------------- #
# Quantum self‑attention (produces a scalar weight per token)
# --------------------------------------------------------------------------- #
class QuantumSelfAttention(nn.Module):
    def __init__(self, n_qubits: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.backend  = qiskit.Aer.get_backend("qasm_simulator")
        self.shots    = 1024

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        seq_len, batch, embed = inputs.shape
        weights = []
        for i in range(seq_len):
            # Use the mean of the token embedding as a rotation angle
            theta = inputs[i].mean().item()
            qc = qiskit.QuantumCircuit(1)
            qc.h(0)
            qc.ry(theta, 0)
            qc.measure_all()
            job = qiskit.execute(qc, self.backend, shots=self.shots)
            result = job.result().get_counts(qc)
            exp = 0.0
            for state, c in result.items():
                z = int(state, 2)
                exp += (1 if z == 0 else -1) * c
            exp /= self.shots
            weights.append(exp)
        weights = torch.tensor(weights, device=inputs.device, dtype=torch.float32)
        weights = weights.unsqueeze(-1).expand(-1, batch, embed)
        return inputs * weights

# --------------------------------------------------------------------------- #
# Quantum LSTM cell
# --------------------------------------------------------------------------- #
class QuantumQLayer(nn.Module):
    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        self.backend  = qiskit.Aer.get_backend("aer_simulator")
        self.shots    = 1024

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Map input tensor to rotation angles
        angles = x.detach().cpu().numpy().flatten()
        qc = qiskit.QuantumCircuit(self.n_wires)
        for i, angle in enumerate(angles):
            qc.rx(angle, i)
        for i in range(self.n_wires - 1):
            qc.cx(i, i + 1)
        qc.measure_all()
        job = qiskit.execute(qc, self.backend, shots=self.shots)
        result = job.result().get_counts(qc)
        exp = 0.0
        for state, c in result.items():
            z = int(state, 2)
            exp += (1 if z == 0 else -1) * c
        return torch.tensor([exp / self.shots], device=x.device, dtype=torch.float32)

class QuantumQLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits   = n_qubits

        self.forget = QuantumQLayer(n_qubits)
        self.input = QuantumQLayer(n_qubits)
        self.update = QuantumQLayer(n_qubits)
        self.output = QuantumQLayer(n_qubits)

        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input  = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
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

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

# --------------------------------------------------------------------------- #
# Quantum fully‑connected layer
# --------------------------------------------------------------------------- #
class QuantumFCL(nn.Module):
    def __init__(self, n_qubits: int, backend, shots: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.backend   = backend
        self.shots     = shots
        self.circuit  = qiskit.QuantumCircuit(n_qubits)
        self.theta    = qiskit.circuit.Parameter("theta")
        self.circuit.h(range(n_qubits))
        self.circuit.barrier()
        self.circuit.ry(self.theta, range(n_qubits))
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts(self.circuit)

        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array(list(count_dict.keys())).astype(float)
            probabilities = counts / self.shots
            return np.sum(states * probabilities)

        if isinstance(result, list):
            return np.array([expectation(item) for item in result])
        return np.array([expectation(result)])

    def forward(self, thetas: torch.Tensor) -> torch.Tensor:
        thetas_np = thetas.detach().cpu().numpy().flatten()
        expectation = self.run(thetas_np)
        return torch.tensor(expectation, device=thetas.device, dtype=torch.float32)

# --------------------------------------------------------------------------- #
# Quantum hybrid head
# --------------------------------------------------------------------------- #
class QuantumHybridFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumFCL, shift: float):
        ctx.shift = shift
        ctx.circuit = circuit
        expectation_z = ctx.circuit.run(inputs.tolist())
        result = torch.tensor(expectation_z, device=inputs.device, dtype=torch.float32)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        input_values = inputs.detach().cpu().numpy().flatten()
        shift = np.ones_like(input_values) * ctx.shift
        gradients = []
        for idx, value in enumerate(input_values):
            right = ctx.circuit.run([value + shift[idx]])
            left  = ctx.circuit.run([value - shift[idx]])
            gradients.append(right - left)
        gradients = torch.tensor(gradients, device=inputs.device, dtype=torch.float32)
        return gradients * grad_output, None, None

class QuantumHybrid(nn.Module):
    def __init__(self, n_qubits: int, backend, shots: int, shift: float):
        super().__init__()
        self.quantum_circuit = QuantumFCL(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return QuantumHybridFunction.apply(inputs, self.quantum_circuit, self.shift)

# --------------------------------------------------------------------------- #
# Main hybrid classifier
# --------------------------------------------------------------------------- #
class HybridSelfAttentionQLSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = QuantumSelfAttention(n_qubits)
        self.lstm      = QuantumQLSTM(embed_dim, hidden_dim, n_qubits)
        self.fcl       = QuantumFCL(n_qubits, qiskit.Aer.get_backend("qasm_simulator"), shots=100)
        self.hybrid    = QuantumHybrid(n_qubits, qiskit.Aer.get_backend("qasm_simulator"),
                                       shots=100, shift=np.pi / 2)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        # sentence: (seq_len, batch)
        x = self.embedding(sentence)                     # (seq_len, batch, embed_dim)
        attn_out = self.attention(x)                     # (seq_len, batch, embed_dim)
        lstm_out, _ = self.lstm(attn_out)                # (seq_len, batch, hidden_dim)
        lstm_avg = lstm_out.mean(dim=0)                  # (batch, hidden_dim)
        fcl_out  = self.fcl(lstm_avg)                    # (batch, 1)
        logits   = self.hybrid(fcl_out)                   # (batch, 1)
        return torch.cat((logits, 1 - logits), dim=-1)    # (batch, 2)

__all__ = ["HybridSelfAttentionQLSTMClassifier"]
