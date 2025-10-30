"""Quantum implementations for the hybrid LSTM architecture.

The module defines:
  * ``QuantumConv`` – a 2×2 quantum filter that returns a probability
    value.
  * ``QuantumSelfAttention`` – a quantum circuit that implements a
    self‑attention style block and outputs a probability vector.
  * ``QuantumLayer`` – reusable variational block used inside the
    quantum LSTM gates.
  * ``QLSTM`` – a quantum‑gated LSTM cell where each gate is realised by
    a small quantum circuit.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer
from qiskit.circuit.random import random_circuit


class QuantumConv:
    """Quantum 2×2 convolution filter.

    Accepts a 2‑D array of shape ``(2, 2)`` and returns a scalar
    probability value wrapped as a ``torch.Tensor``.
    """
    def __init__(self, kernel_size: int = 2):
        self.n_qubits = kernel_size ** 2
        self.circuit = QuantumCircuit(self.n_qubits, self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f'theta{i}') for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self.circuit.rx(self.theta[i], i)
        self.circuit.barrier()
        self.circuit += random_circuit(self.n_qubits, 2)
        self.circuit.measure(range(self.n_qubits), range(self.n_qubits))
        self.backend = Aer.get_backend('qasm_simulator')
        self.shots = 100
        self.threshold = 127

    def run(self, data: np.ndarray) -> torch.Tensor:
        data = data.reshape(1, self.n_qubits)
        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)
        job = execute(self.circuit, self.backend, shots=self.shots,
                      parameter_binds=param_binds)
        result = job.result().get_counts(self.circuit)
        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val
        prob = counts / (self.shots * self.n_qubits)
        return torch.tensor(prob, dtype=torch.float32)


class QuantumSelfAttention:
    """Quantum self‑attention block that returns a probability
    vector representing the measurement outcomes of a small circuit.
    """
    def __init__(self, n_qubits: int = 4):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, 'q')
        self.cr = ClassicalRegister(n_qubits, 'c')
        self.backend = Aer.get_backend('qasm_simulator')

    def _build_circuit(self, rotation_params: np.ndarray,
                       entangle_params: np.ndarray) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(self, rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            inputs: np.ndarray) -> np.ndarray:
        circuit = self._build_circuit(rotation_params, entangle_params)
        # For a lightweight demo we ignore ``inputs`` and just run the circuit.
        job = execute(circuit, self.backend, shots=1024)
        result = job.result().get_counts(circuit)
        probs = np.zeros(2 ** self.n_qubits)
        for state, count in result.items():
            idx = int(state[::-1], 2)  # reverse bit order
            probs[idx] = count
        probs /= probs.sum()
        return probs


class QuantumLayer(tq.QuantumModule):
    """Reusable variational block used inside the quantum LSTM gates."""
    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
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
            tgt = 0 if wire == self.n_wires - 1 else wire + 1
            tqf.cnot(qdev, wires=[wire, tgt])
        return self.measure(qdev)


class QLSTM(nn.Module):
    """Quantum‑gated LSTM cell where each gate is realised by a small
    variational circuit defined in :class:`QuantumLayer`."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        self.forget = QuantumLayer(n_qubits)
        self.input = QuantumLayer(n_qubits)
        self.update = QuantumLayer(n_qubits)
        self.output = QuantumLayer(n_qubits)

        # Linear projections to qubit space
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(self, inputs: torch.Tensor,
                states: tuple[torch.Tensor, torch.Tensor] | None = None):
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
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(self, inputs: torch.Tensor,
                     states: tuple[torch.Tensor, torch.Tensor] | None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

__all__ = ["QLSTM", "QuantumConv", "QuantumSelfAttention"]
