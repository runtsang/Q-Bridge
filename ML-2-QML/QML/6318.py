"""Quantum module providing quantum gates for an LSTM cell and a quantum expectation head.

The module defines:
- `QuantumCircuitGate`: a parametrised circuit executing rotations and CNOTs.
- `QuantumExpectationHead`: a variational circuit that maps a scalar to a probability.
- `QuantumLSTMCellWrapper`: a PyTorch module that wraps the quantum LSTM gates and performs a forward pass.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import qiskit
from qiskit import assemble, transpile
from qiskit.circuit import Parameter
from qiskit.providers.aer import AerSimulator

class QuantumCircuitGate(nn.Module):
    """Single‑qubit rotation followed by a CNOT to another qubit."""
    def __init__(self, n_qubits: int = 4, backend=None, shots: int = 1024):
        super().__init__()
        self.n_qubits = n_qubits
        self.backend = backend or AerSimulator()
        self.shots = shots
        self.theta = Parameter("theta")

        self.circuit = qiskit.QuantumCircuit(n_qubits)
        # apply RX rotation on all qubits
        for q in range(n_qubits):
            self.circuit.rx(self.theta, q)
        # entangle with CNOT chain
        for q in range(n_qubits - 1):
            self.circuit.cx(q, q + 1)
        # measure all qubits
        self.circuit.barrier()
        self.circuit.measure_all()

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        # angles: (batch, n_qubits)
        batch = angles.shape[0]
        expectations = []
        for i in range(batch):
            angle = angles[i]
            compiled = transpile(self.circuit, self.backend)
            param_binds = [{self.theta: float(angle[j])} for j in range(self.n_qubits)]
            qobj = assemble(compiled, shots=self.shots, parameter_binds=param_binds)
            job = self.backend.run(qobj)
            result = job.result()
            counts = result.get_counts()
            exp = 0.0
            for state, cnt in counts.items():
                z = 1 if state[-1] == '0' else -1
                exp += z * cnt
            exp /= self.shots
            expectations.append(exp)
        return torch.tensor(expectations, device=angles.device, dtype=torch.float32)

class QuantumExpectationHead(nn.Module):
    """Variational circuit that maps a scalar to a probability via expectation."""
    def __init__(self, n_qubits: int = 2, backend=None, shots: int = 1024):
        super().__init__()
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta = Parameter("theta")
        self.circuit.h(0)
        self.circuit.rx(self.theta, 0)
        self.circuit.measure_all()
        self.backend = backend or AerSimulator()
        self.shots = shots

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        # logits shape (batch, 1)
        angles = logits.squeeze(-1)  # (batch,)
        compiled = transpile(self.circuit, self.backend)
        param_binds = [{self.theta: float(a)} for a in angles.detach().cpu().numpy()]
        qobj = assemble(compiled, shots=self.shots, parameter_binds=param_binds)
        job = self.backend.run(qobj)
        result = job.result()
        counts = result.get_counts()
        exp = 0.0
        for state, cnt in counts.items():
            z = 1 if state[-1] == '0' else -1
            exp += z * cnt
        exp /= self.shots
        return torch.tensor([exp], device=logits.device, dtype=torch.float32)

class QuantumLSTMCellWrapper(nn.Module):
    """PyTorch wrapper that uses quantum gates for LSTM gates."""
    def __init__(self, hidden_dim: int, n_qubits: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.fc_forget = nn.Linear(hidden_dim * 2, n_qubits)
        self.fc_input = nn.Linear(hidden_dim * 2, n_qubits)
        self.fc_update = nn.Linear(hidden_dim * 2, n_qubits)
        self.fc_output = nn.Linear(hidden_dim * 2, n_qubits)

        self.quantum_gate = QuantumCircuitGate(n_qubits=n_qubits)

    def forward(self, inputs: torch.Tensor):
        """
        inputs: (seq_len, batch, hidden_dim)
        """
        seq_len, batch, hidden_dim = inputs.shape
        hx = torch.zeros(batch, hidden_dim, device=inputs.device)
        cx = torch.zeros(batch, hidden_dim, device=inputs.device)
        outputs = []
        for t in range(seq_len):
            x_t = inputs[t]
            combined = torch.cat([x_t, hx], dim=1)
            f = torch.sigmoid(self.quantum_gate(self.fc_forget(combined)))
            i = torch.sigmoid(self.quantum_gate(self.fc_input(combined)))
            g = torch.tanh(self.quantum_gate(self.fc_update(combined)))
            o = torch.sigmoid(self.quantum_gate(self.fc_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

__all__ = ["QuantumCircuitGate", "QuantumExpectationHead", "QuantumLSTMCellWrapper"]
