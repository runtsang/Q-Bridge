import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
from qiskit import QuantumCircuit, Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator as Estimator

# Quantum Quanvolution – 2×2 patch → 4‑qubit block
class QuantumQuanvolution(tq.QuantumModule):
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                data = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, data)
                self.layer(qdev)
                patches.append(self.measure(qdev).view(bsz, 4))
        return torch.cat(patches, dim=1)

# Quantum LSTM – each gate is a small quantum circuit
class QuantumLSTM(nn.Module):
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

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
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

    def forward(self, inputs: torch.Tensor, states: tuple | None = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(self.linear_forget(combined)))
            i = torch.sigmoid(self.input_gate(self.linear_input(combined)))
            g = torch.tanh(self.update(self.linear_update(combined)))
            o = torch.sigmoid(self.output_gate(self.linear_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(self, inputs: torch.Tensor, states: tuple | None = None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

# Main hybrid estimator – quantum quanvolution → quantum LSTM → EstimatorQNN head
class EstimatorQNNHybrid(nn.Module):
    """
    Quantum‑enhanced EstimatorQNN that chains a quantum quanvolution
    feature extractor, a quantum LSTM cell, and a qiskit EstimatorQNN
    for the final regression head.  The circuit is constructed once
    and reused; input features are fed into the first parameter of
    the circuit while the LSTM output serves as the weight parameters.
    """
    def __init__(self, num_qubits: int = 4, num_outputs: int = 1) -> None:
        super().__init__()
        self.quanvolution = QuantumQuanvolution()
        self.lstm = QuantumLSTM(input_dim=4, hidden_dim=4, n_qubits=num_qubits)

        # Build a simple parametric circuit for EstimatorQNN
        self.input_params = [Parameter(f"inp{idx}") for idx in range(num_qubits)]
        self.weight_params = [Parameter(f"w{idx}") for idx in range(num_qubits)]
        qc = QuantumCircuit(num_qubits)
        for idx, p in enumerate(self.input_params):
            qc.ry(p, idx)
        for idx, p in enumerate(self.weight_params):
            qc.rx(p, idx)

        # Observable for the regression output
        self.observable = SparsePauliOp.from_list([("Z" * num_qubits, 1)])

        # EstimatorQNN head
        self.estimator = Estimator()
        self.estimator_qnn = EstimatorQNN(
            circuit=qc,
            observables=[self.observable],
            input_params=self.input_params,
            weight_params=self.weight_params,
            estimator=self.estimator,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantum feature extraction
        qfeat = self.quanvolution(x)
        # Quantum LSTM to produce weight parameters
        lstm_out, _ = self.lstm(qfeat.unsqueeze(0))  # (1, batch, hidden)
        # Aggregate LSTM output to a vector of weights
        weight_vals = lstm_out.squeeze(0).mean(dim=0)  # (hidden,)
        # Convert to a tensor matching weight_params length
        weight_tensor = torch.tensor([float(v) for v in weight_vals], device=x.device)
        # Forward through EstimatorQNN
        return self.estimator_qnn(x, weight_tensor)

__all__ = ["EstimatorQNNHybrid"]
