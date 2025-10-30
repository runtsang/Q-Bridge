import torch
import torch.nn as nn
import torchquantum as tq
from torchquantum.functional import func_name_dict
from QLSTM import QLSTM as QuantumQLSTM

class FraudQLayer(tq.QuantumModule):
    """Quantum gate module that emulates a fraud‑detection style layer with
    a small parameterised circuit followed by classical scaling and shifting."""
    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        # encode each input feature into a rotation
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
        # classical parameters
        self.scale = nn.Parameter(torch.ones(n_wires))
        self.shift = nn.Parameter(torch.zeros(n_wires))

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor) -> torch.Tensor:
        self.encoder(q_device, x)
        for wire, gate in enumerate(self.params):
            gate(q_device, wires=wire)
        out = self.measure(q_device)
        return out * self.scale + self.shift

class QuantumKernel(tq.QuantumModule):
    """Fixed Ry ansatz that evaluates a quantum kernel between two vectors."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        # encode x and y with opposite phases
        self.ansatz = [
            {"input_idx": [0], "func": "ry", "wires": [0]},
            {"input_idx": [1], "func": "ry", "wires": [1]},
            {"input_idx": [2], "func": "ry", "wires": [2]},
            {"input_idx": [3], "func": "ry", "wires": [3]},
        ]

    @tq.static_support
    def encode(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        for idx, wire in enumerate(range(self.n_wires)):
            func_name_dict["ry"](q_device, wires=[wire], params=x[:, idx])
        for idx, wire in enumerate(range(self.n_wires)):
            func_name_dict["ry"](q_device, wires=[wire], params=-y[:, idx])

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.encode(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

class HybridQLSTM(QuantumQLSTM):
    """
    Quantum implementation of the hybrid LSTM. Gates are realised by small
    parameterised quantum circuits and the forget gate is modulated by a
    quantum kernel similarity to a set of reference vectors.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int = 0,
        reference_vectors: list[torch.Tensor] | None = None,
    ) -> None:
        super().__init__(input_dim, hidden_dim, n_qubits)
        self.forget_gate = FraudQLayer(n_qubits)
        self.input_gate  = FraudQLayer(n_qubits)
        self.update_gate = FraudQLayer(n_qubits)
        self.output_gate = FraudQLayer(n_qubits)
        self.kernel = QuantumKernel()
        self.reference_vectors = reference_vectors or []

    def _kernel_score(self, combined: torch.Tensor) -> torch.Tensor:
        batch_size = combined.size(0)
        if not self.reference_vectors:
            return torch.ones(batch_size, device=combined.device)
        sims = []
        for i in range(batch_size):
            sims_i = [self.kernel(combined[i:i+1], r.reshape(1, -1)) for r in self.reference_vectors]
            sims.append(torch.mean(torch.stack(sims_i)))
        return torch.stack(sims)

    def forward(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=combined.shape[0], device=combined.device)
            f_raw = self.forget_gate(qdev, combined)
            i_raw = self.input_gate(qdev, combined)
            g_raw = self.update_gate(qdev, combined)
            o_raw = self.output_gate(qdev, combined)
            f = torch.sigmoid(f_raw)
            i = torch.sigmoid(i_raw)
            g = torch.tanh(g_raw)
            o = torch.sigmoid(o_raw)
            sim = self._kernel_score(combined)
            f = f * sim
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

__all__ = ["HybridQLSTM"]
