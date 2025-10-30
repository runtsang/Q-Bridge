"""QuantumKernelCombined: quantum‑centric implementation.

The quantum module mirrors the API of the classical module but
provides quantum‑enhanced versions of the kernel, auto‑encoder,
and LSTM tagger.  It uses torchquantum for the kernel and LSTM,
and qiskit for a simple auto‑encoder sampler.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler as QiskitSampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from typing import Iterable, Tuple, Sequence, List

# --------------------------------------------------------------------------- #
# Quantum RBF kernel
# --------------------------------------------------------------------------- #

class QuantumKernelAnsatz(tq.QuantumModule):
    """Fixed Ry‑encoding followed by a reverse‑encoding for the RBF."""
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.q_device.reset_states(x.shape[0])
        for w in range(self.n_wires):
            tq.RY(self.q_device, wires=[w], params=x[:, w])
        for w in range(self.n_wires):
            tq.RY(self.q_device, wires=[w], params=-y[:, w])
        return torch.abs(self.q_device.states.view(-1)[0])

class QuantumKernel(tq.QuantumModule):
    """Quantum kernel that exposes a Gram‑matrix API."""
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.ansatz = QuantumKernelAnsatz(n_wires)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.ansatz(x, y)

    def gram_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        return np.array([[self.forward(x, y).item() for y in b] for x in a])

# --------------------------------------------------------------------------- #
# Quantum auto‑encoder (sampler‑based)
# --------------------------------------------------------------------------- #

def QuantumAutoencoder(num_features: int, latent_dim: int) -> SamplerQNN:
    """
    Builds a simple swap‑test based auto‑encoder.
    The circuit is very small and is meant for demonstration only.
    """
    sampler = QiskitSampler()
    ansatz = RealAmplitudes(num_features, reps=3)

    # Encoder
    qc = QuantumCircuit(num_features)
    qc.append(ansatz, range(num_features))

    # Swap test with an auxiliary qubit
    aux = QuantumRegister(1, "aux")
    qc.add_register(aux)
    qc.h(aux[0])
    for i in range(num_features):
        qc.cswap(aux[0], i, i)
    qc.h(aux[0])
    qc.measure(aux[0], ClassicalRegister(1, "c"))

    # The sampler will produce the probability of measuring |0⟩
    # which we interpret as the reconstruction fidelity.
    return SamplerQNN(circuit=qc,
                      input_params=[],
                      weight_params=[],
                      interpret=lambda x: x,
                      output_shape=(1,),
                      sampler=sampler)

# --------------------------------------------------------------------------- #
# Quantum LSTM cell
# --------------------------------------------------------------------------- #

class QLSTM(nn.Module):
    """LSTM cell where each gate is a small quantum circuit."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [0], "func": "rx", "wires": [0]},
                 {"input_idx": [1], "func": "rx", "wires": [1]},
                 {"input_idx": [2], "func": "rx", "wires": [2]},
                 {"input_idx": [3], "func": "rx", "wires": [3]}]
            )
            self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            for wire in range(self.n_wires):
                tgt = 0 if wire == self.n_wires - 1 else wire + 1
                tqf.cnot(qdev, wires=[wire, tgt])
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

    def forward(self,
                inputs: torch.Tensor,
                states: Tuple[torch.Tensor, torch.Tensor] | None = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs: List[torch.Tensor] = []
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
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

# --------------------------------------------------------------------------- #
# Quantum LSTM tagger
# --------------------------------------------------------------------------- #

class LSTMTagger(nn.Module):
    """Sequence tagger that can use either a classical LSTM or the quantum LSTM."""
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 n_qubits: int = 0):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=False)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out)
        return torch.log_softmax(tag_logits, dim=-1)

# --------------------------------------------------------------------------- #
# Composite class
# --------------------------------------------------------------------------- #

class QuantumKernelCombined:
    """
    Unified interface that exposes both classical and quantum back‑ends.

    Parameters
    ----------
    use_quantum_kernel : bool
        If True, the kernel will be the quantum implementation.
    use_quantum_lstm : bool
        If True, the LSTM tagger will use the quantum LSTM cell.
    use_quantum_autoencoder : bool
        If True, the auto‑encoder will be a quantum sampler‑based circuit.
    """
    def __init__(self,
                 use_quantum_kernel: bool = True,
                 use_quantum_lstm: bool = True,
                 use_quantum_autoencoder: bool = True,
                 kernel_n_wires: int = 4,
                 autoencoder_features: int = 5,
                 autoencoder_latent: int = 3,
                 lstm_params: Tuple[int, int, int, int] | None = None):
        # Kernel
        self.kernel = QuantumKernel(n_wires=kernel_n_wires) if use_quantum_kernel else ClassicalKernel(gamma=1.0)

        # Auto‑encoder
        if use_quantum_autoencoder:
            self.autoencoder = QuantumAutoencoder(autoencoder_features, autoencoder_latent)
        else:
            # Fallback to a simple classical auto‑encoder
            cfg = AutoencoderConfig(input_dim=autoencoder_features)
            self.autoencoder = AutoencoderNet(cfg)

        # LSTM tagger
        if lstm_params is None:
            lstm_params = (50, 100, 5000, 10)
        emb_dim, hid_dim, vocab_sz, tag_sz = lstm_params
        self.tagger = LSTMTagger(emb_dim, hid_dim, vocab_sz, tag_sz,
                                 n_qubits=kernel_n_wires if use_quantum_lstm else 0)

    # ----------------------------------------------------------------------- #
    # Kernel utilities
    # ----------------------------------------------------------------------- #
    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Return the Gram matrix of the chosen kernel."""
        return self.kernel.gram_matrix(a, b)

    # ----------------------------------------------------------------------- #
    # Auto‑encoder utilities
    # ----------------------------------------------------------------------- #
    def train_autoencoder(self,
                          data: torch.Tensor,
                          epochs: int = 10,
                          batch_size: int = 32,
                          lr: float = 1e-3) -> List[float]:
        """
        If the auto‑encoder is classical, run a standard training loop.
        For the quantum sampler the function simply returns a list of dummy
        fidelities because training is performed on a classical optimizer
        outside this module.
        """
        if hasattr(self.autoencoder, "forward"):
            # Classical path
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.autoencoder.to(device)
            dataset = TensorDataset(data)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=lr)
            loss_fn = nn.MSELoss()
            history: List[float] = []

            for _ in range(epochs):
                epoch_loss = 0.0
                for batch, in loader:
                    batch = batch.to(device)
                    optimizer.zero_grad()
                    recon = self.autoencoder(batch)
                    loss = loss_fn(recon, batch)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item() * batch.size(0)
                history.append(epoch_loss / len(dataset))
            return history
        else:
            # Quantum path – return placeholder fidelity history
            return [0.0] * epochs

    # ----------------------------------------------------------------------- #
    # LSTM utilities
    # ----------------------------------------------------------------------- #
    def tag_sequence(self, sentence: torch.Tensor) -> torch.Tensor:
        """Run the tagger on a single sentence."""
        return self.tagger(sentence)

__all__ = ["QuantumKernelCombined", "QuantumKernel", "QLSTM", "LSTMTagger"]
