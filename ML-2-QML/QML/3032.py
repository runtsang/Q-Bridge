import torch
import torch.nn as nn
import torch.quantum as tq
import torch.quantum.functional as tqf
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from typing import Iterable, Tuple, List

def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """Construct a data‑uploading variational circuit."""
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return circuit, list(encoding), list(weights), observables

class QLSTM(nn.Module):
    """Quantum‑enhanced LSTM cell using torchquantum."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [0], "func": "rx", "wires": [0]},
                 {"input_idx": [1], "func": "rx", "wires": [1]},
                 {"input_idx": [2], "func": "rx", "wires": [2]},
                 {"input_idx": [3], "func": "rx", "wires": [3]}]
            )
            self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
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

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
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

    def _init_states(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

__all__ = ["build_classifier_circuit", "QLSTM"]
