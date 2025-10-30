"""Quantum hybrid LSTM with optional QFCModel feature extractor.

The implementation follows the quantum‑LSTM design from the seed
and uses torchquantum for the gate layers.  It also reuses the
QFCModel from the Quantum‑NAT reference as a feature extractor.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from typing import Tuple, Iterable, List, Sequence

class HybridQLSTMQuantum(tq.QuantumModule):
    """Quantum‑hybrid LSTM that can be used as a drop‑in replacement for
    :class:`HybridQLSTM`.  The recurrent cell gates are realised by small
    quantum circuits, while the feature extraction stage is a
    classical‑quantum CNN (QFCModel).  The class is fully differentiable
    and works with the autograd engine of PyTorch.

    Parameters
    ----------
    input_dim : int
        Dimension of the input embedding vector.
    hidden_dim : int
        Hidden state size.
    n_qubits : int
        Number of qubits used for the quantum gates.
    """

    class QLayer(tq.QuantumModule):
        """Single quantum gate layer used for each LSTM gate."""
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
            qdev = tq.QuantumDevice(n_wires=self.n_wires,
                                    bsz=x.shape[0],
                                    device=x.device)
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            # linear‑like entanglement pattern
            for wire in range(self.n_wires):
                tgt = 0 if wire == self.n_wires - 1 else wire + 1
                tqf.cnot(qdev, wires=[wire, tgt])
            return self.measure(qdev)

    class QFCModel(tq.QuantumModule):
        """Quantum fully‑connected feature extractor from the Quantum‑NAT paper."""
        def __init__(self) -> None:
            super().__init__()
            self.n_wires = 4
            self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
            self.q_layer = HybridQLSTMQuantum.QLayer(self.n_wires)
            self.measure = tq.MeasureAll(tq.PauliZ)
            self.norm = nn.BatchNorm1d(self.n_wires)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            bsz = x.shape[0]
            qdev = tq.QuantumDevice(n_wires=self.n_wires,
                                    bsz=bsz,
                                    device=x.device,
                                    record_op=True)
            pooled = F.avg_pool2d(x, 6).view(bsz, 16)
            self.encoder(qdev, pooled)
            self.q_layer(qdev)
            out = self.measure(qdev)
            return self.norm(out)

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 n_qubits: int,
                 use_qfc: bool = False,
                 qfc_model: tq.QuantumModule | None = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.use_qfc = use_qfc

        # LSTM gate circuits
        self.forget = self.QLayer(n_qubits)
        self.input = self.QLayer(n_qubits)
        self.update = self.QLayer(n_qubits)
        self.output = self.QLayer(n_qubits)

        # linear projections to match qubit count
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

        # optional quantum feature extractor
        if use_qfc:
            self.qfc = qfc_model or self.QFCModel()
        else:
            self.qfc = None

        self.hidden2tag = nn.Linear(hidden_dim, 4)  # arbitrary tag size

    def forward(self,
                inputs: torch.Tensor,
                states: Tuple[torch.Tensor, torch.Tensor] | None = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the quantum LSTM.
        ``inputs`` is expected to be of shape (seq_len, batch, feature).
        """
        if self.qfc is not None:
            # feature extraction on the first input token
            # reshape to (batch, 1, channels, H, W) as in QFCModel
            batch = inputs.size(1)
            seq_len = inputs.size(0)
            feats = inputs.view(seq_len, batch, -1)
            # simple reshape, assuming the feature vector can be mapped to 1x1 image
            feats = feats.view(seq_len, batch, 1, 1, feats.size(-1))
            feats = self.qfc(feats.squeeze(2))  # (batch, n_wires)
            inputs = feats.unsqueeze(0)  # make seq_len=1 for simplicity
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

    def _init_states(self,
                     inputs: torch.Tensor,
                     states: Tuple[torch.Tensor, torch.Tensor] | None) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

    # ------------------------------------------------------------------
    # Fast estimator utilities – adapted to quantum circuits
    # ------------------------------------------------------------------
    def evaluate(self,
                 observables: Iterable[tq.PauliZ | tq.PauliX | tq.PauliY],
                 parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        """
        Evaluate expectation values of quantum observables for each
        parameter set.  The method uses a temporary quantum circuit
        that is built from the current parameters of the LSTM gates.
        """
        # This is a toy illustration – a realistic implementation
        # would build a full circuit from the gate parameters.
        results: List[List[complex]] = []
        for params in parameter_sets:
            row = [0 + 0j for _ in observables]
            results.append(row)
        return results
