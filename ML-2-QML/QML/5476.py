"""Quantum‑enhanced hybrid LSTM module using torchquantum.

The implementation follows the same public API as the classical module but
replaces the recurrent layer with a quantum‑parameterised LSTM cell.  It
also includes a QCNN‑style quantum feature extractor and a quantum sampler.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


# --------------------------------------------------------------------------- #
#  Quantum layer used by the LSTM gates
# --------------------------------------------------------------------------- #
class QLayer(tq.QuantumModule):
    """Small quantum circuit that maps a classical vector to a quantum state."""

    def __init__(self, n_wires: int) -> None:
        super().__init__()
        self.n_wires = n_wires
        # Encode classical data into qubit rotations
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "rx", "wires": [i]}
                for i in range(n_wires)
            ]
        )
        # Trainable rotation gates
        self.params = nn.ModuleList(
            [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
        self.encoder(qdev, x)
        for wire, gate in enumerate(self.params):
            gate(qdev, wires=wire)
        # Entangle the qubits
        for wire in range(self.n_wires - 1):
            tqf.cnot(qdev, wires=[wire, wire + 1])
        return self.measure(qdev)


# --------------------------------------------------------------------------- #
#  Quantum LSTM cell
# --------------------------------------------------------------------------- #
class QLSTM(nn.Module):
    """LSTM cell where each gate is implemented by a small quantum circuit."""

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Quantum gates for each LSTM gate
        self.forget = QLayer(n_qubits)
        self.input = QLayer(n_qubits)
        self.update = QLayer(n_qubits)
        self.output = QLayer(n_qubits)

        # Classical linear layers that feed the quantum gates
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(
        self, inputs: torch.Tensor, states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
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
        return torch.cat(outputs, dim=0), (hx, cx)

    def _init_states(
        self, inputs: torch.Tensor, states: Optional[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )


# --------------------------------------------------------------------------- #
#  Quantum QCNN‑style feature extractor
# --------------------------------------------------------------------------- #
class QCNNQuantum(tq.QuantumModule):
    """QCNN‑style feature extractor implemented with torchquantum."""

    def __init__(self, num_wires: int) -> None:
        super().__init__()
        self.num_wires = num_wires
        # Encode the input vector
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        # Random layer + rotations
        self.q_layer = QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.num_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)


# --------------------------------------------------------------------------- #
#  Quantum sampler
# --------------------------------------------------------------------------- #
class SamplerModule(tq.QuantumModule):
    """Quantum sampler that outputs a probability distribution."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["2xRy"])
        self.q_layer = QLayer(2)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(2, 2)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=2, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return F.softmax(self.head(features), dim=-1)


# --------------------------------------------------------------------------- #
#  Hybrid LSTM (classical or quantum)
# --------------------------------------------------------------------------- #
class HybridQLSTM(nn.Module):
    """Hybrid LSTM that can run in classical or quantum mode.

    Parameters
    ----------
    input_dim : int
        Dimensionality of each input token.
    hidden_dim : int
        Hidden state dimensionality.
    n_qubits : int, default 0
        If zero, a classical :class:`torch.nn.LSTM` is used.
    task : str, default 'tagging'
        Either ``'tagging'`` or ``'regression'``.
    tagset_size : int, default 10
        Number of tags for the tagging task.
    use_qcnet : bool, default False
        Whether to apply the QCNN feature extractor before the LSTM.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int = 0,
        task: str = "tagging",
        tagset_size: int = 10,
        use_qcnet: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.task = task

        # Choose the recurrent layer
        if n_qubits == 0:
            self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        else:
            self.lstm = QLSTM(input_dim, hidden_dim, n_qubits)

        # Output head
        out_dim = 1 if task == "regression" else tagset_size
        self.head = nn.Linear(hidden_dim, out_dim)

        # Optional QCNN quantum feature extractor
        self.qcnet: Optional[QCNNQuantum] = QCNNQuantum(n_qubits or input_dim) if use_qcnet else None

        # Quantum sampler
        self.sampler = SamplerModule()

    # ----------------------------------------------------------------------- #
    #  Forward pass
    # ----------------------------------------------------------------------- #
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.task == "tagging":
            lstm_out, _ = self.lstm(inputs)
            logits = self.head(lstm_out)
            return F.log_softmax(logits, dim=-1)

        if self.task == "regression":
            features = self.qcnet(inputs) if self.qcnet else inputs
            features = features.unsqueeze(1)  # sequence length 1
            lstm_out, _ = self.lstm(features)
            return self.head(lstm_out.squeeze(1))

        raise ValueError(f"Unsupported task: {self.task}")

    # ----------------------------------------------------------------------- #
    #  Sampler interface
    # ----------------------------------------------------------------------- #
    def sample(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return a probability distribution from the quantum sampler."""
        return self.sampler(inputs)


__all__ = ["HybridQLSTM", "QCNNQuantum", "SamplerModule"]
