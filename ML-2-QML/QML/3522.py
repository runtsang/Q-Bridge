"""Quantum‑enhanced hybrid LSTM tagger with a quantum convolutional filter.

The implementation follows the same API as the classical
`HybridLSTMTagger` but replaces the gate logic with small
parameterised qiskit circuits and the pre‑processing step with a
quantum quanv filter.  The class can operate in two modes:
* `n_qubits > 0` – quantum gates and quantum conv filter are used.
* `n_qubits == 0` – falls back to a purely classical LSTM
  equivalent to the original `HybridLSTMTagger`.

The module is self‑contained, depends only on `qiskit`,
`numpy` and `torch`, and can be used as a drop‑in replacement
for the classical tagger in experiments that require quantum
variational layers.

"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Quantum convolutional filter (from Conv.py)
# --------------------------------------------------------------------------- #
class QuantumConv:
    """A simple quanv filter that maps a 2‑D array to a scalar."""
    def __init__(self, kernel_size: int, threshold: float = 0.0,
                 shots: int = 1024) -> None:
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.shots = shots
        self.backend = Aer.get_backend("qasm_simulator")

        self.circuit = QuantumCircuit(self.n_qubits)
        self.theta = [Parameter(f"theta{i}") for i in range(self.n_qubits)]
        # Encode data via RX rotations
        for i in range(self.n_qubits):
            self.circuit.rx(self.theta[i], i)
        self.circuit.barrier()
        # Add a simple entangling layer
        for i in range(self.n_qubits - 1):
            self.circuit.cx(i, i + 1)
        self.circuit.measure_all()

    def run(self, data: np.ndarray) -> float:
        """Run the filter on a 2‑D array of shape (k, k)."""
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for valvec in data:
            bind = {self.theta[i]: np.pi if valvec[i] > self.threshold else 0
                    for i in range(self.n_qubits)}
            param_binds.append(bind)

        job = execute(self.circuit,
                      self.backend,
                      shots=self.shots,
                      parameter_binds=param_binds)
        result = job.result()
        counts = result.get_counts(self.circuit)

        # Compute average probability of measuring |1> across all qubits
        total_ones = 0
        total_shots = 0
        for bitstring, freq in counts.items():
            ones = bitstring.count('1')
            total_ones += ones * freq
            total_shots += freq
        return total_ones / (total_shots * self.n_qubits)

# --------------------------------------------------------------------------- #
# Quantum gate (variational circuit)
# --------------------------------------------------------------------------- #
class QuantumGate:
    """Small parameterised quantum circuit that outputs a real value."""
    def __init__(self, n_qubits: int, shots: int = 512) -> None:
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = Aer.get_backend("qasm_simulator")

        # Create parameterised circuit
        self.circuit = QuantumCircuit(n_qubits)
        self.theta = [Parameter(f"theta{i}") for i in range(n_qubits)]
        for i in range(n_qubits):
            self.circuit.rx(self.theta[i], i)
        self.circuit.barrier()
        for i in range(n_qubits - 1):
            self.circuit.cx(i, i + 1)
        self.circuit.measure_all()

        # Initialise parameters randomly
        self.params = np.random.uniform(-np.pi, np.pi, size=n_qubits)

    def run(self, x: np.ndarray) -> float:
        """Evaluate the circuit on the input vector `x` (size n_qubits)
        and return the average |1> probability."""
        # Encode data into rotation angles
        bind = {self.theta[i]: np.pi if x[i] > 0.5 else 0
                for i in range(self.n_qubits)}
        job = execute(self.circuit,
                      self.backend,
                      shots=self.shots,
                      parameter_binds=[bind])
        result = job.result()
        counts = result.get_counts(self.circuit)
        total_ones = 0
        total_shots = 0
        for bitstring, freq in counts.items():
            ones = bitstring.count('1')
            total_ones += ones * freq
            total_shots += freq
        return total_ones / (total_shots * self.n_qubits)

# --------------------------------------------------------------------------- #
# Quantum hybrid LSTM cell
# --------------------------------------------------------------------------- #
class HybridQLSTM(nn.Module):
    """LSTM cell that uses quantum gates for its internal logic."""
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 n_qubits: int = 0,
                 conv_kernel: int = 2,
                 conv_threshold: float = 0.0) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        self.conv = QuantumConv(kernel_size=conv_kernel,
                                 threshold=conv_threshold)

        if n_qubits > 0:
            # Quantum gates
            self.forget_gate = QuantumGate(n_qubits)
            self.input_gate = QuantumGate(n_qubits)
            self.update_gate = QuantumGate(n_qubits)
            self.output_gate = QuantumGate(n_qubits)

            # Linear projection from concatenated input/hidden to gate dim
            self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)
        else:
            # Classical LSTM fallback
            self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
            self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
            self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
            self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def _init_states(self,
                     inputs: torch.Tensor,
                     states: Tuple[torch.Tensor, torch.Tensor] | None = None
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

    def forward(self,
                inputs: torch.Tensor,
                states: Tuple[torch.Tensor, torch.Tensor] | None = None
               ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []

        for x in inputs.unbind(dim=0):
            # Quantum conv feature
            conv_feat = self.conv.run(x.cpu().numpy())
            conv_tensor = torch.tensor(conv_feat, device=x.device,
                                       dtype=x.dtype).unsqueeze(0)
            combined = torch.cat([x, conv_tensor], dim=1)

            if self.n_qubits > 0:
                # Map to gate dimension
                f_raw = self.linear_forget(combined).cpu().numpy()
                i_raw = self.linear_input(combined).cpu().numpy()
                g_raw = self.linear_update(combined).cpu().numpy()
                o_raw = self.linear_output(combined).cpu().numpy()

                f = torch.sigmoid(torch.tensor(
                    self.forget_gate.run(f_raw), dtype=x.dtype, device=x.device))
                i = torch.sigmoid(torch.tensor(
                    self.input_gate.run(i_raw), dtype=x.dtype, device=x.device))
                g = torch.tanh(torch.tensor(
                    self.update_gate.run(g_raw), dtype=x.dtype, device=x.device))
                o = torch.sigmoid(torch.tensor(
                    self.output_gate.run(o_raw), dtype=x.dtype, device=x.device))
            else:
                f = torch.sigmoid(self.forget_linear(combined))
                i = torch.sigmoid(self.input_linear(combined))
                g = torch.tanh(self.update_linear(combined))
                o = torch.sigmoid(self.output_linear(combined))

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

# --------------------------------------------------------------------------- #
# Quantum hybrid LSTM tagger
# --------------------------------------------------------------------------- #
class HybridLSTMTagger(nn.Module):
    """Sequence tagging using a quantum hybrid LSTM backbone."""
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 n_qubits: int = 0,
                 conv_kernel: int = 2,
                 conv_threshold: float = 0.0) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = HybridQLSTM(embedding_dim + 1,
                                    hidden_dim,
                                    n_qubits=n_qubits,
                                    conv_kernel=conv_kernel,
                                    conv_threshold=conv_threshold)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["HybridQLSTM", "HybridLSTMTagger"]
