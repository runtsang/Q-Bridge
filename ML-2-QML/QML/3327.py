"""Quantum‑enhanced LSTM with optional quantum convolution.

The `HybridQLSTM` class below mirrors the classical implementation
but replaces the linear gates with small variational circuits
implemented in torchquantum.  A quantum quanvolution filter is
provided via a qiskit circuit that measures the average |1> probability.
Both components expose the same public API as the anchor module,
allowing seamless switching between classical, quantum, or hybrid
behaviours.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np
import qiskit
from qiskit.circuit.random import random_circuit
import torchquantum as tq
import torchquantum.functional as tqf

class _QuanvCircuit:
    """Filter circuit used for quanvolution layers."""
    def __init__(self, kernel_size: int, backend, shots: int, threshold: float):
        self.n_qubits = kernel_size ** 2
        self._circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()

        self.backend = backend
        self.shots = shots
        self.threshold = threshold

    def run(self, data: np.ndarray) -> float:
        # data shape: (kernel, kernel)
        data = data.reshape(1, self.n_qubits)
        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)

        job = qiskit.execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result().get_counts(self._circuit)

        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val

        return counts / (self.shots * self.n_qubits)

class QuantumConvWrapper(nn.Module):
    """Wraps the qiskit quanv circuit into a torch‑compatible module."""
    def __init__(self, input_dim: int, kernel_size: int = 2, shots: int = 100, threshold: float = 127):
        super().__init__()
        self.input_dim = input_dim
        self.kernel_size = kernel_size
        backend = qiskit.Aer.get_backend("qasm_simulator")
        self.filter = _QuanvCircuit(kernel_size, backend, shots, threshold)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, input_dim)
        batch, seq_len, _ = x.shape
        out = []
        for t in range(seq_len):
            slice_ = x[:, t, :].reshape(batch, self.kernel_size, self.kernel_size)
            conv_t = []
            for b in range(batch):
                conv_t.append(self.filter.run(slice_[b].cpu().numpy()))
            out.append(torch.tensor(conv_t, dtype=x.dtype, device=x.device))
        conv_out = torch.stack(out, dim=1).unsqueeze(-1)  # (batch, seq_len, 1)
        return conv_out

class HybridQLSTM(nn.Module):
    """Quantum‑enhanced LSTM cell with optional quantum convolution."""
    class _QLayer(tq.QuantumModule):
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
            self.params = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
            )
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

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int = 0,
        conv_type: str = "none",
        conv_kernel: int = 2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.conv_type = conv_type
        self.conv_kernel = conv_kernel

        # Optional quantum convolution
        if conv_type == "quantum":
            self.conv = QuantumConvWrapper(input_dim, kernel_size=conv_kernel)
        else:
            self.conv = None

        # Core LSTM gates
        if n_qubits > 0:
            assert n_qubits == hidden_dim, "For quantum mode, n_qubits must equal hidden_dim"
            self.forget = self._QLayer(n_qubits)
            self.input  = self._QLayer(n_qubits)
            self.update = self._QLayer(n_qubits)
            self.output = self._QLayer(n_qubits)

            self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.linear_input  = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)
        else:
            # Classical linear gates
            self.linear_forget = nn.Linear(input_dim + hidden_dim, hidden_dim)
            self.linear_input  = nn.Linear(input_dim + hidden_dim, hidden_dim)
            self.linear_update = nn.Linear(input_dim + hidden_dim, hidden_dim)
            self.linear_output = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def _apply_conv(self, seq: torch.Tensor) -> torch.Tensor:
        """Apply the quantum convolution to each time‑step."""
        if self.conv is None:
            return seq
        conv_out = self.conv(seq)  # (batch, seq_len, 1)
        conv_out = conv_out.expand(-1, -1, self.input_dim)
        return seq + conv_out

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        inputs = self._apply_conv(inputs)
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=1):
            combined = torch.cat([x, hx], dim=1)
            if self.n_qubits > 0:
                f = torch.sigmoid(self.forget(self.linear_forget(combined)))
                i = torch.sigmoid(self.input(self.linear_input(combined)))
                g = torch.tanh(self.update(self.linear_update(combined)))
                o = torch.sigmoid(self.output(self.linear_output(combined)))
            else:
                f = torch.sigmoid(self.linear_forget(combined))
                i = torch.sigmoid(self.linear_input(combined))
                g = torch.tanh(self.linear_update(combined))
                o = torch.sigmoid(self.linear_output(combined))

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(1))
        outputs = torch.cat(outputs, dim=1)
        return outputs, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(0)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

class LSTMTagger(nn.Module):
    """Sequence tagging model that can switch between classical and quantum LSTM."""
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        conv_type: str = "none",
        conv_kernel: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = HybridQLSTM(
            embedding_dim,
            hidden_dim,
            n_qubits=n_qubits,
            conv_type=conv_type,
            conv_kernel=conv_kernel,
        )
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=-1)

__all__ = ["HybridQLSTM", "LSTMTagger"]
