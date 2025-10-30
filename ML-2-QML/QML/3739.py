import torch
from torch import nn
from typing import Tuple
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import StatevectorEstimator
import torchquantum as tq
import torchquantum.functional as tqf

class QLSTMQuantum(nn.Module):
    """
    Quantum LSTM cell where gates are realised by small quantum circuits.
    Adapted from the TorchQuantum QLSTM implementation.
    """
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
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
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            for wire in range(self.n_wires - 1):
                tqf.cnot(qdev, wires=[wire, wire + 1])
            tqf.cnot(qdev, wires=[self.n_wires - 1, 0])
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

    def forward(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None = None):
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
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

class HybridEstimator(nn.Module):
    """
    Quantum‑enhanced hybrid estimator that uses a Qiskit EstimatorQNN for static
    features and a TorchQuantum QLSTM for sequence tagging. The design mirrors
    the classical HybridEstimator but replaces the feed‑forward and LSTM modules
    with quantum counterparts.
    """
    def __init__(self, static_input_dim: int = 1, seq_input_dim: int = 2,
                 hidden_dim: int = 8, tagset_size: int = 10, n_qubits: int = 8) -> None:
        super().__init__()
        # Build Qiskit circuit for static part
        self._build_qiskit_estimator(static_input_dim)
        # Quantum LSTM for sequence part
        self.sequence = QLSTMQuantum(seq_input_dim, hidden_dim, n_qubits=n_qubits)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def _build_qiskit_estimator(self, static_input_dim: int):
        params = [Parameter(f"inp{i}") for i in range(static_input_dim)]
        weight_params = [Parameter(f"w{i}") for i in range(static_input_dim)]
        qc = QuantumCircuit(static_input_dim)
        qc.h(0)
        for i in range(static_input_dim):
            qc.ry(params[i], i)
            qc.rx(weight_params[i], i)
        observable = SparsePauliOp.from_list([("Y" * static_input_dim, 1)])
        estimator = StatevectorEstimator()
        self.estimator_qnn = QiskitEstimatorQNN(
            circuit=qc,
            observables=observable,
            input_params=params,
            weight_params=weight_params,
            estimator=estimator,
        )

    def forward(self, static_inputs: torch.Tensor, seq_inputs: torch.Tensor):
        # Static part: evaluate Qiskit EstimatorQNN
        static_np = static_inputs.detach().cpu().numpy()
        static_out_np = self.estimator_qnn.forward(static_np)
        static_out = torch.tensor(static_out_np, device=static_inputs.device, dtype=static_inputs.dtype)
        # Sequence part
        seq = seq_inputs.transpose(0, 1)  # (seq_len, batch, feat)
        lstm_out, _ = self.sequence(seq)
        tag_logits = self.hidden2tag(lstm_out)
        return static_out, tag_logits

def EstimatorQNN():
    """
    Factory function retained for backward compatibility. Returns an instance of
    HybridEstimator with default parameters, mimicking the original EstimatorQNN
    signature.
    """
    return HybridEstimator()
