import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from typing import Tuple, Optional

class QLayer(tq.QuantumModule):
    """
    Variational quantum circuit that produces a probability distribution
    over a single qubit.  Supports tunable noise and a choice of backend.
    """
    def __init__(self,
                 n_qubits: int,
                 backend: str = "default.qubit",
                 noise_strength: float = 0.0) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.backend = backend
        self.noise_strength = noise_strength

        # Parameterised rotation layer
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "rx", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "rz", "wires": [2]},
                {"input_idx": [3], "func": "rx", "wires": [3]},
            ]
        )
        self.params = nn.ModuleList(
            [tq.RX(has_params=True, trainable=True) for _ in range(n_qubits)]
        )

        # Measurement
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, n_qubits)
        qdev = tq.QuantumDevice(n_wires=self.n_qubits,
                                bsz=x.shape[0],
                                device=x.device,
                                backend=self.backend)
        # Encode input features
        self.encoder(qdev, x)
        # Apply variational parameters
        for wire, gate in enumerate(self.params):
            gate(qdev, wires=wire)
        # Entangle neighbouring qubits
        for wire in range(self.n_qubits):
            if wire == self.n_qubits - 1:
                tqf.cnot(qdev, wires=[wire, 0])
            else:
                tqf.cnot(qdev, wires=[wire, wire + 1])
        # Optional noise injection
        if self.noise_strength > 0.0:
            qdev.apply_noise(tq.DepolarizingNoise(p=self.noise_strength))
        # Measure
        return self.measure(qdev)

class QLSTM(nn.Module):
    """
    Quantum-enhanced LSTM where each gate is realised as a variational
    quantum circuit.  The hidden state is still classical, but the
    gate logits are sampled from the quantum measurements.
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 n_qubits: int,
                 bottleneck_dim: Optional[int] = None,
                 backend: str = "default.qubit",
                 noise_strength: float = 0.0) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.bottleneck_dim = bottleneck_dim
        self.backend = backend
        self.noise_strength = noise_strength

        # Optional bottleneck projection
        if self.bottleneck_dim is not None:
            self.bottleneck = nn.Linear(self.hidden_dim, self.bottleneck_dim)
            gate_dim = self.bottleneck_dim
        else:
            self.bottleneck = None
            gate_dim = self.hidden_dim

        # Quantum layers for gates
        self.forget_gate = QLayer(n_qubits, backend=backend, noise_strength=noise_strength)
        self.input_gate = QLayer(n_qubits, backend=backend, noise_strength=noise_strength)
        self.update_gate = QLayer(n_qubits, backend=backend, noise_strength=noise_strength)
        self.output_gate = QLayer(n_qubits, backend=backend, noise_strength=noise_strength)

        # Linear projections to quantum space
        self.forget_linear = nn.Linear(input_dim + gate_dim, n_qubits)
        self.input_linear = nn.Linear(input_dim + gate_dim, n_qubits)
        self.update_linear = nn.Linear(input_dim + gate_dim, n_qubits)
        self.output_linear = nn.Linear(input_dim + gate_dim, n_qubits)

    def forward(self,
                inputs: torch.Tensor,
                states: Tuple[torch.Tensor, torch.Tensor] | None = None
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            hx_proj = self.bottleneck(hx) if self.bottleneck is not None else hx
            combined = torch.cat([x, hx_proj], dim=1)
            # Gate logits from quantum circuits
            f = torch.sigmoid(self.forget_gate(self.forget_linear(combined)))
            i = torch.sigmoid(self.input_gate(self.input_linear(combined)))
            g = torch.tanh(self.update_gate(self.update_linear(combined)))
            o = torch.sigmoid(self.output_gate(self.output_linear(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(self,
                     inputs: torch.Tensor,
                     states: Tuple[torch.Tensor, torch.Tensor] | None,
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

class LSTMTagger(nn.Module):
    """
    Sequence tagging model that can switch between the classical bottleneck LSTM
    and the quantum-enhanced LSTM defined above.
    """
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 n_qubits: int = 0,
                 bottleneck_dim: Optional[int] = None,
                 backend: str = "default.qubit",
                 noise_strength: float = 0.0) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim,
                              n_qubits=n_qubits,
                              bottleneck_dim=bottleneck_dim,
                              backend=backend,
                              noise_strength=noise_strength)
        else:
            self.lstm = QLSTM(embedding_dim, hidden_dim,
                              n_qubits=0,
                              bottleneck_dim=bottleneck_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTM", "LSTMTagger"]
