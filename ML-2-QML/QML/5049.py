from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# quantum imports
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.algorithms import Sampler as QiskitSampler
import torchquantum as tq
import torchquantum.functional as tqf

__all__ = ["SamplerQNN"]

class SamplerQNN(nn.Module):
    """
    Quantum‑enhanced SamplerQNN that mirrors the classical implementation
    but replaces the sampler, quanvolution and LSTM with quantum circuits.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        tagset_size: int = 17,
        n_qubits: int = 4,
    ) -> None:
        super().__init__()
        self.sampler = self._build_sampler()
        self.quanv = self._build_quanv()
        self.lstm = self._build_lstm(hidden_dim, n_qubits)
        self.tag_head = nn.Linear(hidden_dim, tagset_size)

    # --------------------------------------------------------------------- #
    #  1. Quantum sampler
    # --------------------------------------------------------------------- #
    def _build_sampler(self) -> nn.Module:
        """
        Builds a small parameterised circuit on 2 qubits that outputs the
        probability of the |00⟩ state. The circuit is wrapped in a
        PyTorch module that accepts a batch of 2‑dim inputs.
        """
        inputs = ParameterVector("input", 2)
        weights = ParameterVector("weight", 4)

        qc = QuantumCircuit(2)
        qc.ry(inputs[0], 0)
        qc.ry(inputs[1], 1)
        qc.cx(0, 1)
        qc.ry(weights[0], 0)
        qc.ry(weights[1], 1)
        qc.cx(0, 1)
        qc.ry(weights[2], 0)
        qc.ry(weights[3], 1)

        sampler = QiskitSampler()

        class SamplerWrapper(nn.Module):
            def __init__(self, qc: QuantumCircuit, sampler: QiskitSampler) -> None:
                super().__init__()
                self.qc = qc
                self.sampler = sampler

            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                """
                Args:
                    inputs: tensor of shape (batch, 2)

                Returns:
                    prob00: tensor of shape (batch,)
                """
                probs = []
                for batch in inputs:
                    param_binds = {
                        "input[0]": float(batch[0]),
                        "input[1]": float(batch[1]),
                        "weight[0]": 0.5,
                        "weight[1]": 0.5,
                        "weight[2]": 0.5,
                        "weight[3]": 0.5,
                    }
                    result = self.sampler.run(self.qc.bind_parameters(param_binds))
                    counts = result.get_counts()
                    prob_00 = counts.get("00", 0) / sum(counts.values())
                    probs.append(prob_00)
                return torch.tensor(probs, device=inputs.device)

        return SamplerWrapper(qc, sampler)

    # --------------------------------------------------------------------- #
    #  2. Quantum quanvolution
    # --------------------------------------------------------------------- #
    def _build_quanv(self) -> nn.Module:
        """
        Implements the quantum quanvolution filter from the reference
        using torchquantum. The filter operates on 2×2 image patches.
        """

        class Quanv(tq.QuantumModule):
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
                self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
                self.measure = tq.MeasureAll(tq.PauliZ)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                """
                Args:
                    x: image tensor of shape (batch, 1, 28, 28)

                Returns:
                    flattened feature vector of shape (batch, 4*14*14)
                """
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

        return Quanv()

    # --------------------------------------------------------------------- #
    #  3. Quantum LSTM
    # --------------------------------------------------------------------- #
    def _build_lstm(self, hidden_dim: int, n_qubits: int) -> nn.Module:
        """
        Builds a quantum‑enhanced LSTM cell where each gate is a small
        parameterised quantum circuit. The hidden state is stored classically.
        """

        class QLayer(tq.QuantumModule):
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
                qdev = tq.QuantumDevice(
                    n_wires=self.n_wires, bsz=x.shape[0], device=x.device
                )
                self.encoder(qdev, x)
                for wire, gate in enumerate(self.params):
                    gate(qdev, wires=wire)
                for wire in range(self.n_wires):
                    if wire == self.n_wires - 1:
                        tqf.cnot(qdev, wires=[wire, 0])
                    else:
                        tqf.cnot(qdev, wires=[wire, wire + 1])
                return self.measure(qdev)

        class QLSTM(nn.Module):
            def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
                super().__init__()
                self.input_dim = input_dim
                self.hidden_dim = hidden_dim
                self.n_qubits = n_qubits

                self.forget = QLayer(n_qubits)
                self.input = QLayer(n_qubits)
                self.update = QLayer(n_qubits)
                self.output = QLayer(n_qubits)

                self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
                self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
                self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
                self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

            def forward(
                self,
                inputs: torch.Tensor,
                states: tuple[torch.Tensor, torch.Tensor] | None = None,
            ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
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

            def _init_states(
                self,
                inputs: torch.Tensor,
                states: tuple[torch.Tensor, torch.Tensor] | None = None,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                if states is not None:
                    return states
                batch_size = inputs.size(1)
                device = inputs.device
                return (
                    torch.zeros(batch_size, self.hidden_dim, device=device),
                    torch.zeros(batch_size, self.hidden_dim, device=device),
                )

        return QLSTM(hidden_dim, hidden_dim, n_qubits)

    # --------------------------------------------------------------------- #
    #  Forward pass
    # --------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: image tensor of shape (batch, 1, 28, 28)

        Returns:
            log‑softmax logits of shape (batch, tagset_size)
        """
        # 1. Extract quantum features
        features = self.quanv(x)

        # 2. Feed into quantum LSTM (seq_len=1)
        lstm_out, _ = self.lstm(features.unsqueeze(1))
        lstm_out = lstm_out.squeeze(1)

        # 3. Tag projection
        logits = self.tag_head(lstm_out)
        return F.log_softmax(logits, dim=-1)
