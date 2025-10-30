"""Quantum hybrid classifier/tagger."""
from __future__ import annotations

from typing import Iterable, Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.providers.aer import AerSimulator

class HybridClassifier(nn.Module):
    """
    Quantum hybrid classifier/tagger.
    For classification: uses a variational quantum circuit.
    For sequence tagging: uses a quantum LSTM cell (QLayer).
    """
    def __init__(self,
                 num_qubits: int,
                 depth: int,
                 hidden_dim: int = 128,
                 vocab_size: Optional[int] = None,
                 tagset_size: Optional[int] = None,
                 use_lstm: bool = False):
        super().__init__()
        self.num_qubits = num_qubits
        self.depth = depth
        self.use_lstm = use_lstm
        if use_lstm:
            assert vocab_size is not None and tagset_size is not None, \
                "vocab_size and tagset_size must be provided when use_lstm=True"
            self.embedding = nn.Embedding(vocab_size, num_qubits)
            self.lstm = self.QLSTM(num_qubits, hidden_dim)
            self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
            self.task = 'tagging'
        else:
            self._build_quantum_circuit()
            self.task = 'classification'

    def _build_quantum_circuit(self):
        """Construct a simple variational circuit used for classification."""
        self.encoding = ParameterVector("x", self.num_qubits)
        self.weights = ParameterVector("theta", self.num_qubits * self.depth)
        self.circuit = QuantumCircuit(self.num_qubits)
        # Data encoding
        for qubit in range(self.num_qubits):
            self.circuit.rx(self.encoding[qubit], qubit)
        # Variational layers
        for _ in range(self.depth):
            for qubit in range(self.num_qubits):
                self.circuit.ry(self.weights[qubit], qubit)
            for qubit in range(self.num_qubits - 1):
                self.circuit.cz(qubit, qubit + 1)
        # Observables for measurement (Z on each qubit)
        self.observables = [SparsePauliOp("I" * i + "Z" + "I" * (self.num_qubits - i - 1))
                            for i in range(self.num_qubits)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.task == 'classification':
            # x shape: (batch, num_qubits)
            param_dict = {str(p): val for p, val in zip(self.encoding, x.t())}
            simulator = AerSimulator()
            bound_circuit = self.circuit.bind_parameters(param_dict)
            job = simulator.run(bound_circuit, shots=1024)
            result = job.result()
            counts = result.get_counts()
            # Convert counts to expectation values
            exp_vals = []
            for op in self.observables:
                exp_vals.append(self._expectation_from_counts(counts, op))
            exp_tensor = torch.tensor(exp_vals, dtype=torch.float32, device=x.device)
            return exp_tensor
        else:
            # Sequence tagging using quantum LSTM
            embeds = self.embedding(x)
            lstm_out, _ = self.lstm(embeds)
            logits = self.hidden2tag(lstm_out)
            return F.log_softmax(logits, dim=-1)

    @staticmethod
    def _expectation_from_counts(counts: dict, op: SparsePauliOp) -> float:
        """Compute expectation value of a Pauli operator from counts."""
        exp = 0.0
        total = 0
        for bitstring, cnt in counts.items():
            val = 1
            for i, bit in enumerate(bitstring[::-1]):
                if op.paulis[i] == 'Z' and bit == '1':
                    val *= -1
            exp += val * cnt
            total += cnt
        return exp / total if total > 0 else 0.0

    class QLayer(tq.QuantumModule):
        """Quantum layer used in the quantum LSTM."""
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
            self.encoder(qdev, x)
            for gate in self.params:
                gate(qdev)
            for i in range(self.n_wires - 1):
                tqf.cnot(qdev, wires=[i, i + 1])
            tqf.cnot(qdev, wires=[self.n_wires - 1, 0])
            return self.measure(qdev)

    class QLSTM(nn.Module):
        """Quantum LSTM cell using QLayer gates."""
        def __init__(self, n_qubits: int, hidden_dim: int):
            super().__init__()
            self.n_qubits = n_qubits
            self.hidden_dim = hidden_dim
            self.forget = self.QLayer(n_qubits)
            self.input = self.QLayer(n_qubits)
            self.update = self.QLayer(n_qubits)
            self.output = self.QLayer(n_qubits)
            self.linear_forget = nn.Linear(n_qubits, hidden_dim)
            self.linear_input = nn.Linear(n_qubits, hidden_dim)
            self.linear_update = nn.Linear(n_qubits, hidden_dim)
            self.linear_output = nn.Linear(n_qubits, hidden_dim)

        def forward(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] = None):
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

        def _init_states(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] = None):
            if states is not None:
                return states
            batch_size = inputs.size(1)
            device = inputs.device
            return (torch.zeros(batch_size, self.hidden_dim, device=device),
                    torch.zeros(batch_size, self.hidden_dim, device=device))
