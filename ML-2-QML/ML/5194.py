import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import numpy as np
from torch.utils.data import Dataset

# --------------------------------------------------------------------------- #
# Dataset utilities – adapted from QuantumRegression.py
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate states of the form cos(theta)|0…0> + e^{i phi} sin(theta)|1…1>."""
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels.astype(np.float32)

class RegressionDataset(Dataset):
    """TensorDataset for the superposition regression task."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int):
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

# --------------------------------------------------------------------------- #
# Classical feed‑forward baseline – adapted from EstimatorQNN.py
# --------------------------------------------------------------------------- #
class ClassicalRegressor(nn.Module):
    def __init__(self, input_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# --------------------------------------------------------------------------- #
# Quantum feed‑forward block – adapted from QuantumRegression.py (tq)
# --------------------------------------------------------------------------- #
class QuantumRegressionModule(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.random_layer(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)
            return None  # output is captured by measure

    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self.QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

# --------------------------------------------------------------------------- #
# Sampler network – adapted from SamplerQNN.py
# --------------------------------------------------------------------------- #
def SamplerQNN() -> nn.Module:
    class SamplerModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 4),
                nn.Tanh(),
                nn.Linear(4, 2),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return F.softmax(self.net(x), dim=-1)
    return SamplerModule()

# --------------------------------------------------------------------------- #
# LSTM / QLSTM tagger – adapted from QLSTM.py
# --------------------------------------------------------------------------- #
class ClassicalLSTMTagger(nn.Module):
    """Standard LSTM for sequence tagging."""
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int, tagset_size: int):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=-1)

class QuantumLSTMTagger(nn.Module):
    """Quantum‑enhanced LSTM cell (QLSTM) – simplified from QLSTM.py."""
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
            self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            for wire in range(self.n_wires):
                if wire == self.n_wires - 1:
                    tq.cnot(qdev, wires=[wire, 0])
                else:
                    tq.cnot(qdev, wires=[wire, wire + 1])
            return self.measure(qdev)

    def __init__(self, embedding_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = embedding_dim
        self.forget = self.QLayer(n_qubits)
        self.input = self.QLayer(n_qubits)
        self.update = self.QLayer(n_qubits)
        self.output = self.QLayer(n_qubits)

        self.linear_forget = nn.Linear(self.input_dim + self.hidden_dim, n_qubits)
        self.linear_input = nn.Linear(self.input_dim + self.hidden_dim, n_qubits)
        self.linear_update = nn.Linear(self.input_dim + self.hidden_dim, n_qubits)
        self.linear_output = nn.Linear(self.input_dim + self.hidden_dim, n_qubits)

    def forward(self, inputs: torch.Tensor, states: tuple[torch.Tensor, torch.Tensor] | None = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=1):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(self.linear_forget(combined)))
            i = torch.sigmoid(self.input(self.linear_input(combined)))
            g = torch.tanh(self.update(self.linear_update(combined)))
            o = torch.sigmoid(self.output(self.linear_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(1))
        stacked = torch.cat(outputs, dim=1)
        return stacked, (hx, cx)

    def _init_states(self, inputs: torch.Tensor, states: tuple[torch.Tensor, torch.Tensor] | None):
        if states is not None:
            return states
        batch_size = inputs.size(0)
        device = inputs.device
        return torch.zeros(batch_size, self.hidden_dim, device=device), torch.zeros(batch_size, self.hidden_dim, device=device)

# --------------------------------------------------------------------------- #
# Hybrid estimator – the core of the new module
# --------------------------------------------------------------------------- #
class HybridEstimatorQNN(nn.Module):
    """
    A drop‑in estimator that can operate in four distinct modes:
    * ``classical`` – simple feed‑forward regression.
    * ``quantum``   – variational quantum circuit (tq) with a classical head.
    * ``sampler``   – probability sampler returning a softmax distribution.
    * ``lstm``      – either classical or quantum LSTM for sequence tagging.
    """
    def __init__(
        self,
        mode: str = "classical",
        input_dim: int = 2,
        hidden_dim: int = 8,
        num_wires: int = 2,
        n_qubits: int = 0,
    ):
        super().__init__()
        self.mode = mode.lower()
        if self.mode == "classical":
            self.model = ClassicalRegressor(input_dim)
        elif self.mode == "quantum":
            self.model = QuantumRegressionModule(num_wires)
        elif self.mode == "sampler":
            self.model = SamplerQNN()
        elif self.mode == "lstm":
            if n_qubits > 0:
                self.model = QuantumLSTMTagger(input_dim, hidden_dim, n_qubits)
            else:
                self.model = ClassicalLSTMTagger(input_dim, hidden_dim, vocab_size=input_dim, tagset_size=1)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def switch_mode(self, mode: str):
        """Change the internal model on the fly."""
        self.__init__(mode=mode)

__all__ = [
    "HybridEstimatorQNN",
    "RegressionDataset",
    "generate_superposition_data",
    "SamplerQNN",
    "ClassicalLSTMTagger",
    "QuantumLSTMTagger",
]
