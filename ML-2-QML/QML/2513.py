import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, AerSimulator
import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf

class QSelfAttention:
    """Quantum self‑attention block implemented with Qiskit."""
    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim
        self.n_qubits = embed_dim
        self.qr = QuantumRegister(self.n_qubits, "q")
        self.cr = ClassicalRegister(self.n_qubits, "c")
        self.backend = AerSimulator()
        # Trainable parameters
        self.rotation_params = nn.Parameter(torch.randn(self.n_qubits * 3))
        self.entangle_params = nn.Parameter(torch.ones(self.n_qubits - 1))

    def _build_circuit(self, rotation_params, entangle_params):
        circuit = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(self, rotation_params, entangle_params, shots=1024):
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = self.backend.run(circuit, shots=shots)
        result = job.result()
        counts = result.get_counts(circuit)
        probs = np.zeros(2 ** self.n_qubits)
        for bitstring, count in counts.items():
            idx = int(bitstring[::-1], 2)
            probs[idx] = count / shots
        return probs

class QLSTMQuantum(nn.Module):
    """Quantum‑enhanced LSTM cell using TorchQuantum."""
    class _QLayer(tq.QuantumModule):
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
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            for wire in range(self.n_wires - 1):
                tqf.cnot(qdev, wires=[wire, wire + 1])
            tqf.cnot(qdev, wires=[self.n_wires - 1, 0])
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        self.forget = self._QLayer(n_qubits)
        self.input_gate = self._QLayer(n_qubits)
        self.update = self._QLayer(n_qubits)
        self.output_gate = self._QLayer(n_qubits)

        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(self, inputs: torch.Tensor, states: tuple = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=1):
            combined = torch.cat([x, hx], dim=-1)
            f = torch.sigmoid(self.forget(self.linear_forget(combined)))
            i = torch.sigmoid(self.input_gate(self.linear_input(combined)))
            g = torch.tanh(self.update(self.linear_update(combined)))
            o = torch.sigmoid(self.output_gate(self.linear_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(1))
        out_seq = torch.cat(outputs, dim=1)
        return out_seq, (hx, cx)

    def _init_states(self, inputs: torch.Tensor, states: tuple = None):
        if states is not None:
            return states
        batch_size = inputs.size(0)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

__all__ = ["QSelfAttention", "QLSTMQuantum"]
