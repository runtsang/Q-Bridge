"""Quantum hybrid binary classifier with optional quantum auto‑encoder and quantum LSTM.

This module mirrors the classical API but replaces the final head with a
parameterised quantum circuit executed on Aer.  A lightweight quantum
auto‑encoder (swap‑test RealAmplitudes) can compress the feature vector
before classification.  The quantum LSTM tagger uses a small circuit per
gate, enabling experimentation on sequence data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import QuantumCircuit, transpile, assemble
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Statevector, SparsePauliOp
import torchquantum as tq
import torchquantum.functional as tqf
from typing import Tuple, Iterable, List

# --------------------------------------------------------------------------- #
#  Classical helpers (re‑implemented for compatibility)                      #
# --------------------------------------------------------------------------- #
def build_classifier_circuit(num_features: int, depth: int = 2) -> nn.Sequential:
    """Same interface as the classical helper."""
    layers: List[nn.Module] = []
    in_dim = num_features
    for _ in range(depth):
        layers.append(nn.Linear(in_dim, num_features))
        layers.append(nn.ReLU(inplace=True))
        in_dim = num_features
    layers.append(nn.Linear(in_dim, 2))
    return nn.Sequential(*layers)

class AutoencoderConfig:
    def __init__(self, input_dim: int, latent_dim: int = 32,
                 hidden_dims: Tuple[int, int] = (128, 64),
                 dropout: float = 0.1):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout

class AutoencoderNet(nn.Module):
    def __init__(self, config: AutoencoderConfig):
        super().__init__()
        encoder_layers = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU(inplace=True))
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU(inplace=True))
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

def Autoencoder(input_dim: int, *, latent_dim: int = 32,
                hidden_dims: Tuple[int, int] = (128, 64),
                dropout: float = 0.1) -> AutoencoderNet:
    cfg = AutoencoderConfig(input_dim, latent_dim, hidden_dims, dropout)
    return AutoencoderNet(cfg)

# --------------------------------------------------------------------------- #
#  Quantum circuit wrapper (parameterised 1‑qubit circuit)                   #
# --------------------------------------------------------------------------- #
class SimpleQuantumCircuit:
    """A minimal 1‑qubit circuit with a parameterised RY gate."""
    def __init__(self, backend, shots: int = 100):
        self.backend = backend
        self.shots = shots
        self.circuit = QuantumCircuit(1)
        self.theta = qiskit.circuit.Parameter("theta")
        self.circuit.h(0)
        self.circuit.ry(self.theta, 0)
        self.circuit.measure_all()

    def run(self, theta: torch.Tensor) -> torch.Tensor:
        """Return the expectation value of Z on the single qubit."""
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(compiled, shots=self.shots,
                        parameter_binds=[{self.theta: val.item()}])
        job = self.backend.run(qobj)
        result = job.result()
        counts = result.get_counts()
        exp = 0.0
        for state, count in counts.items():
            bit = int(state[0])  # single qubit
            exp += (1 - 2 * bit) * count
        exp /= self.shots
        return torch.tensor([exp], dtype=torch.float32, device=theta.device)

# --------------------------------------------------------------------------- #
#  HybridFunction bridging PyTorch and the quantum circuit                  #
# --------------------------------------------------------------------------- #
class HybridFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: SimpleQuantumCircuit, shift: float = 0.0):
        ctx.shift = shift
        ctx.circuit = circuit
        theta = inputs + shift
        out = ctx.circuit.run(theta)
        ctx.save_for_backward(inputs, out)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        eps = 1e-3
        grad = torch.zeros_like(inputs)
        for i in range(inputs.numel()):
            inp = inputs.clone()
            inp[i] += eps
            pos = ctx.circuit.run(inp + shift)
            inp[i] -= 2 * eps
            neg = ctx.circuit.run(inp + shift)
            grad[i] = (pos - neg) / (2 * eps)
        return grad * grad_output, None, None

# --------------------------------------------------------------------------- #
#  Hybrid binary classifier (quantum head)                                  #
# --------------------------------------------------------------------------- #
class HybridBinaryClassifier(nn.Module):
    """
    CNN backbone → optional quantum auto‑encoder → quantum‑expectation head.
    The head uses a parameterised 1‑qubit circuit executed on Aer.
    """
    def __init__(self,
                 use_autoencoder: bool = False,
                 autoencoder_cfg: dict | None = None,
                 use_quantum_head: bool = True,
                 classifier_depth: int = 2):
        super().__init__()
        # CNN backbone identical to the classical version
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        self.use_autoencoder = use_autoencoder
        if use_autoencoder:
            if autoencoder_cfg is None:
                autoencoder_cfg = {}
            self.autoencoder = Autoencoder(**autoencoder_cfg)

        # quantum head
        backend = qiskit.Aer.get_backend("aer_simulator")
        self.quantum_circuit = SimpleQuantumCircuit(backend, shots=200)
        self.shift = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN feature extraction
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)

        # optional quantum auto‑encoder compression
        if self.use_autoencoder:
            x = self.autoencoder.encode(x)

        # fully‑connected layers
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # quantum expectation head
        probs = HybridFunction.apply(x.squeeze(), self.quantum_circuit, self.shift)

        return torch.cat((probs, 1 - probs), dim=-1)

# --------------------------------------------------------------------------- #
#  Quantum auto‑encoder (swap‑test RealAmplitudes)                           #
# --------------------------------------------------------------------------- #
class QuantumAutoencoder(nn.Module):
    """
    A simple quantum auto‑encoder that maps a high‑dimensional feature vector
    to a low‑dimensional latent space using a RealAmplitudes ansatz and a
    swap‑test for reconstruction fidelity.
    """
    def __init__(self, num_latent: int, num_trash: int = 2):
        super().__init__()
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.total_qubits = num_latent + 2 * num_trash + 1
        self.circuit = QuantumCircuit(self.total_qubits)
        ansatz = RealAmplitudes(num_latent + num_trash, reps=3)
        self.circuit.compose(ansatz, range(0, num_latent + num_trash), inplace=True)
        self.circuit.barrier()
        aux = num_latent + 2 * num_trash
        self.circuit.h(aux)
        for i in range(num_trash):
            self.circuit.cswap(aux, num_latent + i, num_latent + num_trash + i)
        self.circuit.h(aux)
        self.circuit.measure(aux, 0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Not implemented: placeholder for a full quantum simulation.
        return inputs

# --------------------------------------------------------------------------- #
#  Quantum LSTM tagger (inspired by QLSTM.py)                                #
# --------------------------------------------------------------------------- #
class QLSTM(nn.Module):
    """LSTM cell where gates are realised by small quantum circuits."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
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

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
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

    def forward(self, inputs: torch.Tensor,
                states: Tuple[torch.Tensor, torch.Tensor] | None = None
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

    def _init_states(self,
                     inputs: torch.Tensor,
                     states: Tuple[torch.Tensor, torch.Tensor] | None = None
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

class LSTMTagger(nn.Module):
    """Sequence tagging model that can switch between classical and quantum LSTM."""
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 n_qubits: int = 0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = [
    "HybridBinaryClassifier",
    "HybridFunction",
    "Autoencoder",
    "AutoencoderNet",
    "AutoencoderConfig",
    "build_classifier_circuit",
    "QLSTM",
    "LSTMTagger",
    "QuantumAutoencoder",
]
