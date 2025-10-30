import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import StatevectorEstimator
import torchquantum as tq
import torchquantum.functional as tqf

# --------------------------------------------------------------------------- #
# Quantum LSTM cell – copies the TorchQuantum implementation from reference 3
# --------------------------------------------------------------------------- #
class QuantumQLSTM(nn.Module):
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
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

    def forward(self, inputs: torch.Tensor, states=None):
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

    def _init_states(self, inputs, states):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

# --------------------------------------------------------------------------- #
# Quantum quanvolution filter – copies the TorchQuantum implementation from reference 4
# --------------------------------------------------------------------------- #
class QuantumQuanvolutionFilter(tq.QuantumModule):
    def __init__(self):
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
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
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
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)

# --------------------------------------------------------------------------- #
# Hybrid class – shares API with the classical version but uses quantum back‑ends
# --------------------------------------------------------------------------- #
class HybridEstimatorQNN(nn.Module):
    """
    Quantum‑centric hybrid estimator.

    Parameters
    ----------
    mode : {"regression", "sequence", "image"}
        Sub‑network to instantiate.
    input_dim, hidden_dim, vocab_size, tagset_size, num_classes
        Hyper‑parameters for the individual sub‑networks.
    """

    def __init__(self,
                 mode: str = "regression",
                 input_dim: int = 2,
                 hidden_dim: int = 8,
                 vocab_size: int = 1000,
                 tagset_size: int = 10,
                 num_classes: int = 10):
        super().__init__()
        self.mode = mode.lower()

        if self.mode == "regression":
            # Variational circuit with two parameters
            params = [Parameter(f"w{i}") for i in range(2)]
            qc = QuantumCircuit(1)
            qc.h(0)
            qc.ry(params[0], 0)
            qc.rx(params[1], 0)

            observable = qiskit.quantum_info.SparsePauliOp.from_list([("Y", 1)])
            estimator = StatevectorEstimator()
            self.estimator_qnn = QiskitEstimatorQNN(
                circuit=qc,
                observables=observable,
                input_params=[params[0]],
                weight_params=[params[1]],
                estimator=estimator,
            )

        elif self.mode == "sequence":
            self.lstm = QuantumQLSTM(input_dim, hidden_dim, n_qubits=hidden_dim)
            self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

        elif self.mode == "image":
            self.qfilter = QuantumQuanvolutionFilter()
            self.fc = nn.Linear(4 * 14 * 14, num_classes)

        else:
            raise ValueError(f"Unknown mode {mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "regression":
            # x shape: (batch, 2) – first column input, second column weight
            inputs = x[:, 0].unsqueeze(1)
            weights = x[:, 1].unsqueeze(1)
            return self.estimator_qnn.predict(inputs, weights).view(-1, 1)

        if self.mode == "sequence":
            lstm_out, _ = self.lstm(x)
            logits = self.hidden2tag(lstm_out)
            return F.log_softmax(logits, dim=-1)

        if self.mode == "image":
            features = self.qfilter(x)
            logits = self.fc(features)
            return F.log_softmax(logits, dim=-1)

        raise RuntimeError("unreachable")

__all__ = ["HybridEstimatorQNN"]
