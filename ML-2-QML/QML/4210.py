import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN as QSamplerQNN
from qiskit.primitives import StatevectorSampler
from qiskit.circuit.library import RealAmplitudes
from typing import Tuple

class QLayer(tq.QuantumModule):
    """Small quantum gate implemented with a parameterised circuit."""
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

class QLSTM(nn.Module):
    """Quantum LSTM cell where gates are realised by small quantum circuits."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.forget = QLayer(n_qubits)
        self.input = QLayer(n_qubits)
        self.update = QLayer(n_qubits)
        self.output = QLayer(n_qubits)
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] = None):
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

    def _init_states(self, inputs, states):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

class QuantumSamplerQNN(nn.Module):
    """Wrapper around Qiskit’s SamplerQNN to expose a torch‑like forward."""
    def __init__(self) -> None:
        super().__init__()
        inputs = ParameterVector("input", 2)
        weights = ParameterVector("weight", 4)
        qc = QuantumCircuit(2)
        qc.ry(inputs[0], 0)
        qc.ry(inputs[1], 1)
        qc.cx(0, 1)
        qc.ry(weights[0], 0)
        qc.ry(weights[1], 1)
        qc.cx(0, 1)
        qc.ry(weights[2], 0)
        qc.ry(weights[3], 1)

        sampler = StatevectorSampler()
        self.qnn = QSamplerQNN(
            circuit=qc,
            input_params=inputs,
            weight_params=weights,
            sampler=sampler,
            interpret=lambda x: x,
            output_shape=2,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inp_np = inputs.detach().cpu().numpy()
        samples = self.qnn.forward(inp_np)
        return torch.tensor(samples, device=inputs.device, dtype=inputs.dtype)

class QuantumAutoencoder(nn.Module):
    """Quantum auto‑encoder that compresses a hidden state to fewer qubits."""
    def __init__(self, num_qubits: int, latent_qubits: int) -> None:
        super().__init__()
        self.num_qubits = num_qubits
        self.latent_qubits = latent_qubits
        self.encoder = RealAmplitudes(num_qubits, reps=3)
        self.decoder = RealAmplitudes(latent_qubits, reps=3)
        self.sampler = StatevectorSampler()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        qdev = tq.QuantumDevice(n_wires=self.num_qubits, bsz=x.shape[0], device=x.device)
        self.encoder(qdev, x)
        probs = self.sampler(qdev)
        return torch.tensor(probs, device=x.device, dtype=x.dtype)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        qdev = tq.QuantumDevice(n_wires=self.latent_qubits, bsz=z.shape[0], device=z.device)
        self.decoder(qdev, z)
        probs = self.sampler(qdev)
        return torch.tensor(probs, device=z.device, dtype=z.dtype)

class HybridQLSTM(nn.Module):
    """Hybrid quantum‑classical LSTM that can optionally use a quantum sampler and auto‑encoder."""
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        use_sampler: bool = False,
        use_autoencoder: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.use_sampler = use_sampler
        self.use_autoencoder = use_autoencoder

        if use_sampler:
            self.sampler = QuantumSamplerQNN()
        if use_autoencoder:
            self.autoencoder = QuantumAutoencoder(num_qubits=embedding_dim, latent_qubits=embedding_dim // 2)

        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)

        if self.use_sampler:
            sampled = self.sampler(embeds[:, :2])
            embeds = torch.cat([embeds, sampled], dim=1)

        if self.use_autoencoder:
            encoded = self.autoencoder.encode(embeds)
            embeds = self.autoencoder.decode(encoded)

        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["HybridQLSTM", "QuantumSamplerQNN", "QuantumAutoencoder"]
