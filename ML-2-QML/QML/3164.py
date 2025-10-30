import torch
from torch import nn
from typing import Tuple
import torchquantum as tq
import torchquantum.functional as tqf

class HybridEstimatorQLSTM(nn.Module):
    """
    Hybrid estimator that combines a classical MLP regressor with a
    quantum‑enhanced LSTM implemented with TorchQuantum.
    """

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # MLP regressor
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        # Quantum LSTM
        self.lstm = QLSTM(input_dim, hidden_dim, n_qubits)

        # Tagging head
        self.tag_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, input_dim).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            reg_output: regression output shape (batch, 1)
            tag_output: tagging logits shape (batch, seq_len, 1)
        """
        # Regression on the first time step
        reg_out = self.regressor(x[:, 0])

        # Sequence tagging
        lstm_out, _ = self.lstm(x)
        tag_logits = self.tag_head(lstm_out)
        return reg_out, tag_logits

class QLSTM(nn.Module):
    """
    Quantum‑augmented LSTM where each gate is implemented by a small
    parameterised quantum circuit. The quantum circuit is built with
    TorchQuantum and produces gate activations via measurement.
    """

    class QGate(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            # Encoder: rotate each wire according to the input vector
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            # Trainable rotation gates for each wire
            self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
            # Measurement
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass of the quantum gate.

            Parameters
            ----------
            x : torch.Tensor
                Input vector of shape (batch, n_wires).

            Returns
            -------
            torch.Tensor
                Measurement outcomes of shape (batch, n_wires).
            """
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
            self.encoder(qdev, x)
            for i, gate in enumerate(self.params):
                gate(qdev, wires=i)
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

        # Quantum gates
        self.forget_gate = self.QGate(n_qubits)
        self.input_gate = self.QGate(n_qubits)
        self.update_gate = self.QGate(n_qubits)
        self.output_gate = self.QGate(n_qubits)

        # Linear projections
        self.forget_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.input_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.update_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.output_lin = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] = None):
        """
        Forward pass of the quantum LSTM.

        Parameters
        ----------
        inputs : torch.Tensor
            Input sequence of shape (batch, seq_len, input_dim).
        states : Tuple[torch.Tensor, torch.Tensor], optional
            Initial hidden and cell states.

        Returns
        -------
        Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            LSTM outputs of shape (batch, seq_len, hidden_dim) and
            final hidden and cell states.
        """
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=1):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_gate(self.forget_lin(combined)))
            i = torch.sigmoid(self.input_gate(self.input_lin(combined)))
            g = torch.tanh(self.update_gate(self.update_lin(combined)))
            o = torch.sigmoid(self.output_gate(self.output_lin(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(1))
        lstm_out = torch.cat(outputs, dim=1)
        return lstm_out, (hx, cx)

    def _init_states(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] = None):
        if states is not None:
            return states
        batch_size = inputs.size(0)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

__all__ = ["HybridEstimatorQLSTM", "QLSTM"]
