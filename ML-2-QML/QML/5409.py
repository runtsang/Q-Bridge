from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit.quantum_info import SparsePauliOp

# ---------- Quantum LSTM ----------
class QLSTMQuantum(nn.Module):
    """
    Variational‑gate LSTM where each gate is a small quantum circuit
    parameterised by trainable rotation angles.
    """
    class QGate(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [0], "func": "ry", "wires": [0]},
                    {"input_idx": [1], "func": "ry", "wires": [1]},
                    {"input_idx": [2], "func": "ry", "wires": [2]},
                    {"input_idx": [3], "func": "ry", "wires": [3]},
                ]
            )
            self.params = nn.ModuleList(
                [tq.RY(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            self.measure(qdev)
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Quantum gates for each LSTM component
        self.forget = self.QGate(n_qubits)
        self.input = self.QGate(n_qubits)
        self.update = self.QGate(n_qubits)
        self.output = self.QGate(n_qubits)

        # Classical linear maps to prepare the quantum state
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(self, inputs: torch.Tensor, states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
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
        return torch.cat(outputs, dim=0), (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

# ---------- Quantum Quanvolution ----------
class QuantumQuanvolutionFilter(tq.QuantumModule):
    """
    Random two‑qubit quantum kernel applied to 2×2 image patches.
    """
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
        self.random_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
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
                    [x[:, r, c], x[:, r, c + 1], x[:, r + 1, c], x[:, r + 1, c + 1]],
                    dim=1,
                )
                self.encoder(qdev, data)
                self.random_layer(qdev)
                patches.append(self.measure(qdev).view(bsz, 4))
        return torch.cat(patches, dim=1)

class QuantumQuanvolutionClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuantumQuanvolutionFilter()
        self.fc = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.qfilter(x)
        logits = self.fc(feat)
        return F.log_softmax(logits, dim=-1)

# ---------- Quantum Estimator ----------
def QuantumEstimatorQNN(input_dim: int = 2, weight_dim: int = 1) -> QiskitEstimatorQNN:
    params = [Parameter("theta")]
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.rx(params[0], 0)
    qc.measure_all()

    observable = SparsePauliOp.from_list([("Z" * qc.num_qubits, 1)])
    estimator = Estimator()
    qnn = QiskitEstimatorQNN(
        circuit=qc,
        observables=observable,
        input_params=[],
        weight_params=params,
        estimator=estimator,
    )
    return qnn

# ---------- HybridQLSTM ----------
class HybridQLSTM(nn.Module):
    """
    Quantum hybrid LSTM that replaces the recurrent core with a
    variational‑gate LSTM, applies a quantum quanvolution feature extractor,
    and finishes with a Qiskit EstimatorQNN head.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int,
        *,
        use_quanvolution: bool = True,
        use_estimator: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Quantum quanvolution
        self.quanvolution = QuantumQuanvolutionFilter() if use_quanvolution else None

        # Quantum LSTM core
        self.lstm = QLSTMQuantum(input_dim, hidden_dim, n_qubits)

        # Quantum estimator head
        self.estimator = QuantumEstimatorQNN() if use_estimator else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expected shape: (batch, seq_len, 1, 28, 28)
        """
        if self.quanvolution is not None:
            b, t, c, h, w = x.shape
            x = x.view(b * t, c, h, w)
            feat = self.quanvolution(x)
            x = feat.view(b, t, -1)

        lstm_out, _ = self.lstm(x)  # quantum core

        if self.estimator is not None:
            lstm_out = self.estimator(lstm_out)

        return lstm_out

__all__ = ["HybridQLSTM", "QuantumQuanvolutionFilter", "QuantumQuanvolutionClassifier", "QuantumEstimatorQNN", "QLSTMQuantum"]
