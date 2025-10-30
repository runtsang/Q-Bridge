from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.circuit import ParameterVector

class QuantumEncoder(nn.Module):
    """Parameterised RX encoding circuit."""
    def __init__(self, num_qubits: int):
        super().__init__()
        self.num_qubits = num_qubits
        self.params = ParameterVector('x', num_qubits)

    def build(self, circuit: QuantumCircuit) -> QuantumCircuit:
        for i, param in enumerate(self.params):
            circuit.rx(param, i)
        return circuit

class QuantumSelfAttention(nn.Module):
    """Small entangling block to emulate self‑attention."""
    def __init__(self, num_qubits: int):
        super().__init__()
        self.num_qubits = num_qubits
        self.params = ParameterVector('theta', num_qubits * 2)

    def build(self, circuit: QuantumCircuit) -> QuantumCircuit:
        for i in range(self.num_qubits):
            circuit.ry(self.params[2*i], i)
        for i in range(self.num_qubits-1):
            circuit.cz(i, i+1)
        return circuit

class QuantumQuanvolution(nn.Module):
    """Two‑qubit kernel applied per patch."""
    def __init__(self, num_qubits: int):
        super().__init__()
        self.num_qubits = num_qubits
        self.params = ParameterVector('q', num_qubits * 4)

    def build(self, circuit: QuantumCircuit) -> QuantumCircuit:
        for i in range(self.num_qubits):
            circuit.rx(self.params[4*i], i)
            circuit.rz(self.params[4*i+1], i)
            circuit.cx(i, (i+1)%self.num_qubits)
        return circuit

class QuantumLSTMCell(nn.Module):
    """Parameterised circuit that mimics an LSTM gate."""
    def __init__(self, num_qubits: int):
        super().__init__()
        self.num_qubits = num_qubits
        self.params = ParameterVector('lstm', num_qubits * 4)

    def build(self, circuit: QuantumCircuit) -> QuantumCircuit:
        for i in range(self.num_qubits):
            circuit.rx(self.params[4*i], i)     # forget
            circuit.ry(self.params[4*i+1], i)   # input
            circuit.rz(self.params[4*i+2], i)   # update
            circuit.cz(i, (i+1)%self.num_qubits)  # output
        return circuit

class HybridQuantumClassifierQML(nn.Module):
    """
    Quantum‑centric version of HybridQuantumClassifier.
    Uses Qiskit to build a parameterised circuit for each sub‑module and
    evaluates it on the Aer simulator.
    """
    def __init__(self, input_dim: int, hidden_dim: int, seq_len: int, num_qubits: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.num_qubits = num_qubits
        self.encoder = QuantumEncoder(num_qubits)
        self.attention = QuantumSelfAttention(num_qubits)
        self.qfilter = QuantumQuanvolution(num_qubits)
        self.lstm_cell = QuantumLSTMCell(num_qubits)
        self.classifier = QuantumEncoder(num_qubits)  # reuse encoder as classifier head
        self.backend = Aer.get_backend('qasm_simulator')
        self.shots = 1024

    def _build_circuit(self, input_vector: np.ndarray) -> QuantumCircuit:
        qr = QuantumRegister(self.num_qubits)
        cr = ClassicalRegister(self.num_qubits)
        circuit = QuantumCircuit(qr, cr)
        # encoding
        for i in range(min(self.num_qubits, len(input_vector))):
            circuit.rx(input_vector[i], i)
        # self‑attention
        circuit = self.attention.build(circuit)
        # quanvolution
        circuit = self.qfilter.build(circuit)
        # LSTM cell
        circuit = self.lstm_cell.build(circuit)
        # classifier head
        circuit = self.classifier.build(circuit)
        circuit.measure(qr, cr)
        return circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.size(0)
        logits = torch.zeros(batch, 2)
        for i in range(batch):
            input_vec = x[i].flatten().cpu().numpy()
            circuit = self._build_circuit(input_vec)
            job = execute(circuit, self.backend, shots=self.shots)
            counts = job.result().get_counts(circuit)
            # Simple mapping: sum of measurement outcomes weighted by counts
            z = 0
            for bitstring, count in counts.items():
                val = sum(1 if b == '1' else -1 for b in bitstring[::-1])
                z += val * count
            logits[i, 0] = z
            logits[i, 1] = -z
        return logits

__all__ = ["HybridQuantumClassifierQML"]
