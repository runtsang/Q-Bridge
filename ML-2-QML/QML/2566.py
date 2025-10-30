"""Quantum‑enhanced LSTM implementation used by UnifiedEstimatorQLSTM.

The QLSTM class below re‑uses the quantum‑classical hybrid design from
the second reference pair.  It is intentionally lightweight so that the
module can be imported in environments that only have the classical
PyTorch stack installed.  When a backend such as TorchQuantum is
available, the quantum gates are instantiated; otherwise the class
falls back to a purely classical linear mapping that mimics the
quantum behaviour.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

# Try to import TorchQuantum; if unavailable, use a dummy implementation.
try:
    import torchquantum as tq
    import torchquantum.functional as tqf
    _HAS_TQ = True
except Exception:  # pragma: no cover
    _HAS_TQ = False


class QLayer(nn.Module):
    """Small quantum module that implements a variational layer.

    When TorchQuantum is present this layer runs a small circuit that
    encodes the input, applies a trainable RX rotation on each wire,
    and performs a chain of CNOTs before measuring in the Z basis.
    If TorchQuantum is not available, the layer simply returns the
    input unchanged, preserving the API for pure‑classical training.
    """
    def __init__(self, n_wires: int) -> None:
        super().__init__()
        self.n_wires = n_wires
        if _HAS_TQ:
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
        else:
            # Dummy placeholders
            self.encoder = None
            self.params = nn.ModuleList()
            self.measure = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not _HAS_TQ:
            # Identity fallback
            return x
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


class QLSTM(nn.Module):
    """LSTM cell where each gate is realised by a small quantum layer.

    The implementation follows the second reference pair but is
    refactored for clarity and optional quantum execution.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Quantum layers for each gate
        self.forget = QLayer(n_qubits)
        self.input = QLayer(n_qubits)
        self.update = QLayer(n_qubits)
        self.output = QLayer(n_qubits)

        # Linear projections to the quantum register
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


__all__ = ["QLSTM"]
