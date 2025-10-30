import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from typing import Tuple, Optional

class QLSTMEnhanced(nn.Module):
    """Quantum‑enhanced LSTM cell with hybrid variational circuits.

    Each gate is realised by a small variational quantum circuit followed
    by a classical feed‑forward network that maps the circuit output to
    the gate activation.  The module also accepts an optional noise model
    for the underlying quantum device and exposes a regularization loss
    that encourages the circuit parameters to stay close to orthogonal
    transformations.
    """
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int, gate_ff_hidden: int,
                     noise_model: Optional[dict] = None):
            super().__init__()
            self.n_wires = n_wires
            self.noise_model = noise_model
            # Encoder that maps the input vector to rotation angles
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [i], "func": "rx", "wires": [i]}
                    for i in range(n_wires)
                ]
            )
            # Classical network to produce rotation angles
            self.ff = nn.Sequential(
                nn.Linear(n_wires, gate_ff_hidden),
                nn.ReLU(),
                nn.Linear(gate_ff_hidden, n_wires),
                nn.Tanh()
            )
            # Trainable rotation gates
            self.params = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x shape: (batch, n_wires)
            angles = self.ff(x)
            qdev = tq.QuantumDevice(n_wires=self.n_wires,
                                    bsz=x.shape[0],
                                    device=x.device,
                                    noise_model=self.noise_model)
            # Encode input as rotations
            self.encoder(qdev, angles)
            # Apply trainable rotations
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            # Entangling gates
            for wire in range(self.n_wires - 1):
                tqf.cnot(qdev, wires=[wire, wire + 1])
            return self.measure(qdev)

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 n_qubits: int,
                 gate_ff_hidden: int = 32,
                 noise_model: Optional[dict] = None,
                 regularizer_weight: float = 0.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        assert hidden_dim == n_qubits, "hidden_dim must equal n_qubits for quantum LSTM."
        self.gate_dim = n_qubits

        # Linear layers that produce raw gate pre‑activations
        self.linear_forget = nn.Linear(input_dim + hidden_dim, self.gate_dim)
        self.linear_input = nn.Linear(input_dim + hidden_dim, self.gate_dim)
        self.linear_update = nn.Linear(input_dim + hidden_dim, self.gate_dim)
        self.linear_output = nn.Linear(input_dim + hidden_dim, self.gate_dim)

        # Quantum layers per gate
        self.forget_gate = self.QLayer(n_qubits, gate_ff_hidden, noise_model)
        self.input_gate = self.QLayer(n_qubits, gate_ff_hidden, noise_model)
        self.update_gate = self.QLayer(n_qubits, gate_ff_hidden, noise_model)
        self.output_gate = self.QLayer(n_qubits, gate_ff_hidden, noise_model)

        self.regularizer_weight = regularizer_weight

    def forward(self,
                inputs: torch.Tensor,
                states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
               ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_gate(self.linear_forget(combined)))
            i = torch.sigmoid(self.input_gate(self.linear_input(combined)))
            g = torch.tanh(self.update_gate(self.linear_update(combined)))
            o = torch.sigmoid(self.output_gate(self.linear_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(self,
                     inputs: torch.Tensor,
                     states: Optional[Tuple[torch.Tensor, torch.Tensor]]
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

    def regularization_loss(self) -> torch.Tensor:
        """Return a penalty that encourages the gate parameters to stay
        close to orthogonal transformations.  The penalty is the sum of
        squared norms of the trainable rotation angles, scaled by
        ``regularizer_weight``.
        """
        loss = 0.0
        for gate in [self.forget_gate, self.input_gate,
                     self.update_gate, self.output_gate]:
            for param_gate in gate.params:
                loss += torch.sum(param_gate.weight**2)
        return self.regularizer_weight * loss

class LSTMTagger(nn.Module):
    """Sequence tagging model that can switch between classical and quantum LSTM."""
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 n_qubits: int = 0,
                 gate_ff_hidden: int = 32,
                 noise_model: Optional[dict] = None,
                 regularizer_weight: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTMEnhanced(embedding_dim,
                                      hidden_dim,
                                      n_qubits=n_qubits,
                                      gate_ff_hidden=gate_ff_hidden,
                                      noise_model=noise_model,
                                      regularizer_weight=regularizer_weight)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTMEnhanced", "LSTMTagger"]
