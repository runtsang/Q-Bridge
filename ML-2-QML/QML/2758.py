from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf

class QuantumSampler(tq.QuantumModule):
    """
    Quantum circuit that encodes a 4‑dimensional feature vector into rotation angles
    and measures Pauli‑Z on all wires.  The raw measurement counts are turned into
    a soft‑max probability distribution that serves as the four LSTM gate values.
    """
    def __init__(self, n_qubits: int = 4) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "rx", "wires": [i]}
                for i in range(n_qubits)
            ]
        )
        self.params = nn.ModuleList(
            [tq.RX(has_params=True, trainable=True) for _ in range(n_qubits)]
        )
        self.cnot_pattern = [(i, (i + 1) % n_qubits) for i in range(n_qubits)]
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=x.shape[0], device=x.device)
        self.encoder(qdev, x)
        for gate, wire in zip(self.params, range(self.n_qubits)):
            gate(qdev, wires=wire)
        for w1, w2 in self.cnot_pattern:
            tqf.cnot(qdev, wires=[w1, w2])
        counts = self.measure(qdev)
        probs = torch.softmax(counts, dim=-1)
        return probs

class SamplerQLSTM(nn.Module):
    """
    Quantum‑enhanced hybrid sampler‑LSTM.
    The hidden state is updated exactly as in a classical LSTM, but the four
    gate probabilities are produced by a small variational quantum circuit.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.hidden_proj = nn.Linear(hidden_dim, hidden_dim)
        self.proj_to_qubits = nn.Linear(hidden_dim, 4)
        self.sampler = QuantumSampler(n_qubits=4)
        self.n_qubits = n_qubits

    def forward(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            x_proj = self.input_proj(x)
            hx_proj = self.hidden_proj(hx)
            combined = x_proj + hx_proj
            qubit_input = self.proj_to_qubits(combined)
            gate_probs = self.sampler(qubit_input)
            f, i, g, o = gate_probs.split(1, dim=-1)
            cx = f * cx + i * torch.tanh(g)
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

__all__ = ["SamplerQLSTM"]
