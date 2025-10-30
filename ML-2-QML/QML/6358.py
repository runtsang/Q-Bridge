import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from typing import Tuple, Optional

class QLSTMGen(QLSTM):
    """
    A hybrid quantum‑classical LSTM with an additional quantum readout
    layer that processes the final hidden state before the tag head.
    The readout uses a parameterised variational circuit that
    outputs a vector of probabilities.  The circuit is trained
    jointly with the rest of the model.
    """

    class QuantumReadout(tq.QuantumModule):
        def __init__(self, n_wires: int, output_dim: int):
            super().__init__()
            self.n_wires = n_wires
            self.output_dim = output_dim
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [0], "func": "rx", "wires": [0]},
                    {"input_idx": [1], "func": "rx", "wires": [1]},
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
            return self.measure(qdev)

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int,
        readout_dim: int = 30,
        **kwargs,
    ):
        super().__init__(input_dim, hidden_dim, n_qubits)
        self.quantum_readout = self.QuantumReadout(n_qubits, readout_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
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

        stacked = torch.cat(outputs, dim=0)

        # Quantum readout on the final hidden state
        q_out = self.quantum_readout(hx)
        q_out = torch.softmax(q_out, dim=-1)
        # Combine classical and quantum outputs
        combined_out = torch.cat([hx, q_out], dim=-1)
        return stacked, (combined_out, cx)
