"""QLSTM__gen007.py – Quantum side of the hybrid LSTM tagger.

This module mirrors the classical API but replaces the LSTM cell with a
quantum‑gate implementation and provides quantum circuits for the
sampler and fully‑connected layer.  The code is fully importable and
provides a `HybridFunction` that forwards through a Qiskit circuit,
allowing gradients to flow via finite‑difference.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the quantum back‑end – we use Qiskit for the sampler and FCL
# and TorchQuantum for the LSTM gates.
import torchquantum as tq
import torchquantum.functional as tqf
from qiskit import QuantumCircuit, Aer, execute, transpile, assemble
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from qiskit.providers.aer import AerSimulator


# --------------------------------------------------------------------------- #
# Quantum LSTM cell – gates are small parametrised circuits
# --------------------------------------------------------------------------- #
class QLSTM(nn.Module):
    """LSTM cell where each gate is a quantum circuit."""

    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires

            # Simple encoding: one RX per input qubit
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [i], "func": "rx", "wires": [i]}
                    for i in range(n_wires)
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
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            # Entangle all qubits with a linear chain of CNOTs
            for wire in range(self.n_wires - 1):
                tqf.cnot(qdev, wires=[wire, wire + 1])
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
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
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )


# --------------------------------------------------------------------------- #
# Quantum sampler – parameterised 2‑qubit circuit
# --------------------------------------------------------------------------- #
class QuantumSampler:
    """Wrapper around a 2‑qubit parameterised circuit executed on Aer."""

    def __init__(self, shots: int = 1024) -> None:
        self.shots = shots
        self.backend = AerSimulator()
        self.circuit = QuantumCircuit(2)
        self.input_params = ParameterVector("input", 2)
        self.weight_params = ParameterVector("weight", 4)

        # Build the circuit once
        self.circuit.ry(self.input_params[0], 0)
        self.circuit.ry(self.input_params[1], 1)
        self.circuit.cx(0, 1)
        self.circuit.ry(self.weight_params[0], 0)
        self.circuit.ry(self.weight_params[1], 1)
        self.circuit.cx(0, 1)
        self.circuit.ry(self.weight_params[2], 0)
        self.circuit.ry(self.weight_params[3], 1)
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """Execute the circuit for the provided angles and return the
        probability of measuring '01' (used as a toy expectation value)."""
        param_binds = [{self.input_params[i]: theta for i, theta in enumerate(thetas)}]
        transpiled = transpile(self.circuit, self.backend)
        qobj = assemble(transpiled, shots=self.shots, parameter_binds=param_binds)
        result = self.backend.run(qobj).result()
        counts = result.get_counts()
        # Convert counts to a single expectation-like value
        probs = np.array(list(counts.values())) / self.shots
        states = np.array(list(counts.keys()), dtype=float)
        return np.sum(states * probs)


# --------------------------------------------------------------------------- #
# Quantum fully‑connected layer – 1‑qubit circuit
# --------------------------------------------------------------------------- #
class QuantumFCL:
    """Parameterised 1‑qubit circuit used as a toy fully‑connected layer."""

    def __init__(self, shots: int = 1024) -> None:
        self.shots = shots
        self.backend = AerSimulator()
        self.circuit = QuantumCircuit(1)
        self.theta = ParameterVector("theta", 1)
        self.circuit.h(0)
        self.circuit.ry(self.theta[0], 0)
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        param_binds = [{self.theta[0]: theta} for theta in thetas]
        transpiled = transpile(self.circuit, self.backend)
        qobj = assemble(transpiled, shots=self.shots, parameter_binds=param_binds)
        result = self.backend.run(qobj).result()
        counts = result.get_counts()
        probs = np.array(list(counts.values())) / self.shots
        states = np.array(list(counts.keys()), dtype=float)
        return np.sum(states * probs)


# --------------------------------------------------------------------------- #
# Hybrid activation – differentiable interface to a quantum circuit
# --------------------------------------------------------------------------- #
class HybridFunction(torch.autograd.Function):
    """Forward pass runs the quantum circuit; backward uses finite‑difference."""

    @staticmethod
    def forward(
        ctx,
        inputs: torch.Tensor,
        circuit: QuantumSampler | QuantumFCL,
        shift: float,
    ) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        # Convert to list for Qiskit
        thetas = inputs.tolist()
        expectation = ctx.circuit.run(np.array(thetas))
        result = torch.tensor([expectation], dtype=torch.float32, device=inputs.device)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = np.ones_like(inputs.numpy()) * ctx.shift
        grads = []
        for idx, value in enumerate(inputs.numpy()):
            right = ctx.circuit.run([value + shift[idx]])
            left = ctx.circuit.run([value - shift[idx]])
            grads.append(right - left)
        grads = torch.tensor(grads, dtype=torch.float32, device=inputs.device)
        return grads * grad_output, None, None


class Hybrid(nn.Module):
    """Wraps a quantum circuit into a PyTorch layer."""

    def __init__(
        self,
        circuit: QuantumSampler | QuantumFCL,
        shift: float = np.pi / 2,
    ) -> None:
        super().__init__()
        self.circuit = circuit
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs, self.circuit, self.shift)


# --------------------------------------------------------------------------- #
# Main tagger – quantum LSTM + optional quantum head
# --------------------------------------------------------------------------- #
class LSTMTagger(nn.Module):
    """Sequence tagging model that can use a quantum LSTM cell and a quantum head."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        use_quantum_head: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        if use_quantum_head:
            # Use the 2‑qubit sampler as the head
            self.head = Hybrid(QuantumSampler(shots=2048), shift=np.pi / 2)
        else:
            self.head = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.unsqueeze(1))
        logits = self.head(lstm_out.squeeze(1))
        if isinstance(self.head, nn.Linear):
            return F.log_softmax(logits, dim=1)
        else:
            # Hybrid head already outputs a probability
            return torch.cat((logits, 1 - logits), dim=-1)


__all__ = [
    "QLSTM",
    "HybridFunction",
    "Hybrid",
    "QuantumSampler",
    "QuantumFCL",
    "LSTMTagger",
]
