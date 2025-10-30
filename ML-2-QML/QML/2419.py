import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class QuantumAutoencoder(tq.QuantumModule):
    """
    Minimal quantum autoencoder that maps an input vector to a small
    set of latent qubits and reconstructs it via a simple ansatz.
    """
    def __init__(self, n_wires: int, latent_dim: int) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.latent_dim = latent_dim
        # Encoder: simple RX rotations parameterised by the input
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "rx", "wires": [i]}
                for i in range(latent_dim)
            ]
        )
        # Decoder: a shallow RealAmplitudes circuit
        self.decoder = tq.RealAmplitudes(n_wires, reps=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
        self.encoder(qdev, x)
        self.decoder(qdev)
        return tqf.measure_all(qdev)

class QLSTM(nn.Module):
    """
    Quantum LSTM cell where each gate is a small quantum circuit.
    """
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [i], "func": "rx", "wires": [i]}
                    for i in range(n_wires)
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

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.forget = self.QLayer(n_qubits)
        self.input = self.QLayer(n_qubits)
        self.update = self.QLayer(n_qubits)
        self.output = self.QLayer(n_qubits)
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
        return torch.cat(outputs, dim=0), (hx, cx)

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

class HybridQLSTMQuantum(nn.Module):
    """
    Quantum hybrid model: compresses embeddings with a quantum autoencoder
    and processes them with a quantum LSTM for sequence tagging.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int,
        latent_dim: int = 3,
        hidden_dims: tuple[int, int] = (128, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.autoencoder = QuantumAutoencoder(n_qubits, latent_dim)
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)                     # (seq_len, batch, embed)
        flat = embeds.view(-1, embeds.size(-1))                     # (seq_len*batch, embed)
        compressed = self.autoencoder(flat)                         # (seq_len*batch, latent)
        compressed = compressed.view(embeds.size(0), embeds.size(1), -1)
        lstm_out, _ = self.lstm(compressed)
        logits = self.hidden2tag(lstm_out)                          # (seq_len, batch, tagset)
        return F.log_softmax(logits, dim=2)

__all__ = ["HybridQLSTMQuantum"]
