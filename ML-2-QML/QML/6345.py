import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from typing import Tuple, Optional

class QuantumEnhancedQLSTM(nn.Module):
    """Quantum LSTM cell with configurable ansatz and measurement regularisation.

    The implementation follows the original QLSTM but adds:
    * A single QLayer that can be reused for all gates.
    * `ansatz_depth` controls how many layers of parameterised rotations
      and entangling gates are stacked.
    * `parameter_sharing` allows all gates to share the same QLayer
      instance, reducing the number of trainable parameters.
    * `measurement_regularisation` adds a penalty on the variance of
      rotation parameters, encouraging exploration during training.
    """

    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int, ansatz_depth: int = 2) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.ansatz_depth = ansatz_depth

            # Encoder that maps classical features to rotation angles
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [i], "func": "rx", "wires": [i]}
                    for i in range(n_wires)
                ]
            )

            # Parameterised rotations for each layer
            self.rotations = nn.ModuleList(
                [
                    tq.RX(has_params=True, trainable=True)
                    for _ in range(ansatz_depth * n_wires)
                ]
            )

            # Measurement of all wires in the Z basis
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
            self.encoder(qdev, x)
            rot_idx = 0
            for _ in range(self.ansatz_depth):
                for wire in range(self.n_wires):
                    self.rotations[rot_idx](qdev, wires=wire)
                    rot_idx += 1
                for wire in range(self.n_wires - 1):
                    tqf.cnot(qdev, wires=[wire, wire + 1])
            return self.measure(qdev)

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int,
        ansatz_depth: int = 2,
        parameter_sharing: bool = True,
        measurement_regularisation: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.ansatz_depth = ansatz_depth
        self.parameter_sharing = parameter_sharing
        self.measurement_regularisation = measurement_regularisation

        # QLayer reused for all gates if parameter_sharing is True
        self.qgate = self.QLayer(n_qubits, ansatz_depth=ansatz_depth)

        # Linear layers to map classical input to qubit space
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.qgate(self.linear_forget(combined)))
            i = torch.sigmoid(self.qgate(self.linear_input(combined)))
            g = torch.tanh(self.qgate(self.linear_update(combined)))
            o = torch.sigmoid(self.qgate(self.linear_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

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

    def measurement_regularisation_loss(self) -> torch.Tensor:
        """Return a penalty on the variance of rotation parameters."""
        if self.measurement_regularisation == 0.0:
            return torch.tensor(0.0, device=self.parameters().__next__().device)
        l2 = torch.tensor(0.0, device=self.parameters().__next__().device)
        for p in self.qgate.parameters():
            l2 = l2 + torch.norm(p, p=2) ** 2
        return self.measurement_regularisation * l2

class LSTMTagger(nn.Module):
    """Sequence tagging model that can switch between classical and quantum LSTM."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        ansatz_depth: int = 2,
        parameter_sharing: bool = True,
        measurement_regularisation: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QuantumEnhancedQLSTM(
                embedding_dim,
                hidden_dim,
                n_qubits=n_qubits,
                ansatz_depth=ansatz_depth,
                parameter_sharing=parameter_sharing,
                measurement_regularisation=measurement_regularisation,
            )
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QuantumEnhancedQLSTM", "LSTMTagger"]
