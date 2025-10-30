import torch
import torch.nn as nn
import torch.nn.functional as F
import strawberryfields as sf
from strawberryfields.ops import Dgate, N
import torchquantum as tq
import torchquantum.functional as tqf
from typing import Tuple

class PhotonicLayer(nn.Module):
    """
    Encode each feature vector into a displacement on a dedicated mode
    and return photon‑number expectations.
    """
    def __init__(self, input_dim: int, n_photons: int = 5) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.n_modes = input_dim
        self.n_photons = n_photons
        self.backend = sf.backends.FockBackend()
        self.backend.set_options(max_fockdim=n_photons + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        outputs = []
        for i in range(batch_size):
            prog = sf.Program(self.n_modes)
            with prog.context as q:
                for idx, val in enumerate(x[i]):
                    Dgate(val.item(), 0) | q[idx]
            result = self.backend.run(prog, shots=1)
            exp = [result.expectation_value(N(idx)) for idx in range(self.n_modes)]
            outputs.append(torch.tensor(exp, dtype=torch.float32))
        return torch.stack(outputs)

class QLSTM(nn.Module):
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

        self.forget = self.QLayer(n_qubits)
        self.input = self.QLayer(n_qubits)
        self.update = self.QLayer(n_qubits)
        self.output = self.QLayer(n_qubits)

        self.linear_forget = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.linear_input = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.linear_update = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.linear_output = nn.Linear(input_dim + hidden_dim, hidden_dim)

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

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

class QuanvolutionFilter(tq.QuantumModule):
    """
    Apply a random two‑qubit quantum kernel to 2×2 image patches.
    """
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

class FraudDetectionHybrid(nn.Module):
    """
    Quantum‑enhanced fraud‑detection architecture that fuses:
    * a Strawberry‑Fields photonic encoder,
    * a quantum LSTM cell for sequential patterns,
    * a quanvolutional filter for image‑style features.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        seq_len: int,
        n_qubits: int = 4,
    ) -> None:
        super().__init__()
        self.photonic = PhotonicLayer(input_dim)
        self.qlstm = QLSTM(input_dim, hidden_dim, n_qubits)
        self.quanv = QuanvolutionFilter()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + 4 * 14 * 14, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Shape (batch, seq_len, input_dim)

        Returns
        -------
        Tensor
            Fraud probability of shape (batch, 1)
        """
        batch_size = x.size(0)
        seq_len = x.size(1)
        flat = x.reshape(batch_size * seq_len, -1)
        photonic_out = self.photonic(flat)
        photonic_out = photonic_out.reshape(batch_size, seq_len, -1)
        qlstm_out, _ = self.qlstm(photonic_out)
        last_hidden = qlstm_out[:, -1, :]
        quanv_out = self.quanv(x)
        combined = torch.cat([last_hidden, quanv_out], dim=1)
        return self.classifier(combined)

__all__ = [
    "PhotonicLayer",
    "QLSTM",
    "QuanvolutionFilter",
    "FraudDetectionHybrid",
]
