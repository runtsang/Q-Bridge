"""
Hybrid quantum LSTM with quantum convolution, variational auto‑encoder
and quantum LSTM gates.

This module mirrors the classical counterpart but replaces the
pre‑processing and recurrent core with quantum circuits.  The
architecture follows a *combination* scaling paradigm: quantum
operations are used where they can provide a potential advantage
while the overall interface remains identical to the classical
implementation.

Key quantum components
----------------------
* `QuanvCircuit` – small quanvolution filter built with Qiskit.
* `QuantumAutoencoder` – sampler‑based variational auto‑encoder.
* `QLayer` – variational circuit that implements an LSTM gate.
* `QLSTM` – LSTM cell with quantum gates for each gate.

The module relies on Qiskit, Pennylane (via `torchquantum`) and
PyTorch.  All components are fully importable and self‑contained.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
import torchquantum as tq
import torchquantum.functional as tqf
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit.random import random_circuit
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.circuit.library import RawFeatureVector
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals

# Import auxiliary modules from the seed codebase
from.Conv import Conv
from.Autoencoder import Autoencoder
from.GraphQNN import random_network, feedforward


# ----------------------------------------------------------------------
# Quantum convolution (QuanvCircuit)
# ----------------------------------------------------------------------
class QuanvCircuit:
    """Small quanvolution filter that encodes a 2×2 patch into a quantum state."""

    def __init__(self, kernel_size: int, backend, shots: int, threshold: float):
        self.n_qubits = kernel_size ** 2
        self._circuit = QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()

        self.backend = backend
        self.shots = shots
        self.threshold = threshold

    def run(self, data: np.ndarray) -> float:
        """Run the quantum circuit on a 2×2 data patch."""
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {
                self.theta[i]: np.pi if val > self.threshold else 0
                for i, val in enumerate(dat)
            }
            param_binds.append(bind)

        job = qiskit.execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result().get_counts(self._circuit)

        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val
        return counts / (self.shots * self.n_qubits)


# ----------------------------------------------------------------------
# Quantum auto‑encoder (SamplerQNN)
# ----------------------------------------------------------------------
def QuantumAutoencoder(num_latent: int = 3, num_trash: int = 2) -> SamplerQNN:
    """Return a sampler‑based variational auto‑encoder."""
    algorithm_globals.random_seed = 42
    sampler = qiskit.primitives.StatevectorSampler()

    def ansatz(num_qubits: int) -> QuantumCircuit:
        return qiskit.circuit.library.RealAmplitudes(num_qubits, reps=5)

    # Build the circuit
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    circuit = QuantumCircuit(qr, cr)
    circuit.compose(ansatz(num_latent + num_trash), range(0, num_latent + num_trash), inplace=True)
    circuit.barrier()
    auxiliary_qubit = num_latent + 2 * num_trash
    circuit.h(auxiliary_qubit)
    for i in range(num_trash):
        circuit.cswap(auxiliary_qubit, num_latent + i, num_latent + num_trash + i)
    circuit.h(auxiliary_qubit)
    circuit.measure(auxiliary_qubit, cr[0])

    def identity_interpret(x):
        return x

    qnn = SamplerQNN(
        circuit=circuit,
        input_params=[],
        weight_params=circuit.parameters,
        interpret=identity_interpret,
        output_shape=2,
        sampler=sampler,
    )
    return qnn


# ----------------------------------------------------------------------
# Quantum LSTM cell (QLayer + QLSTM)
# ----------------------------------------------------------------------
class QLayer(tq.QuantumModule):
    """Variational circuit that implements a single LSTM gate."""

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
            tgt = 0 if wire == self.n_wires - 1 else wire + 1
            tqf.cnot(qdev, wires=[wire, tgt])
        return self.measure(qdev)


class QLSTM(nn.Module):
    """Quantum LSTM cell with variational gates."""

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
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

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


# ----------------------------------------------------------------------
# Hybrid class combining all quantum modules
# ----------------------------------------------------------------------
class HybridQLSTM(nn.Module):
    """
    Hybrid quantum LSTM that applies a quanvolution filter,
    a variational auto‑encoder, and a quantum LSTM core.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        conv_kernel: int = 2,
        autoencoder_latent: int = 3,
        graph_arch: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Quantum convolution
        backend = qiskit.Aer.get_backend("qasm_simulator")
        self.conv = QuanvCircuit(conv_kernel, backend, shots=100, threshold=127)

        # Quantum auto‑encoder
        self.autoencoder = QuantumAutoencoder(autoencoder_latent)

        # Optional graph‑based feature extractor (classical)
        if graph_arch:
            _, self.graph_weights, _, _ = random_network(graph_arch, samples=10)
        else:
            self.graph_weights = None

        # Quantum LSTM core
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _graph_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.graph_weights is None:
            return x
        h = x
        for w in self.graph_weights:
            h = torch.tanh(w @ h)
        return h

    # ------------------------------------------------------------------
    # Core forward pass
    # ------------------------------------------------------------------
    def forward(
        self,
        sentence: torch.Tensor,
    ) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)  # (seq_len, emb_dim)

        # Quantum convolution on the embedding sequence
        # Simulate the 2×2 filter on each embedding vector
        conv_out = []
        for vec in embeds:
            # vec is 1‑D; reshape to 2×2 for the filter
            patch = vec.detach().cpu().numpy().reshape(2, 2)
            conv_val = self.conv.run(patch)
            conv_out.append(torch.tensor(conv_val, device=vec.device))
        conv_out = torch.stack(conv_out)  # (seq_len,)

        # Auto‑encoder bottleneck
        ae_out = self.autoencoder(conv_out.unsqueeze(1))

        # Optional graph feature extraction
        graph_out = self._graph_features(ae_out.squeeze(1))

        # LSTM step
        lstm_out, _ = self.lstm(graph_out.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


class LSTMTagger(nn.Module):
    """
    Sequence tagging model that uses :class:`HybridQLSTM` or the vanilla
    :class:`nn.LSTM` as the recurrent core.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        conv_kernel: int = 2,
        autoencoder_latent: int = 3,
        graph_arch: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        if n_qubits > 0:
            self.lstm = HybridQLSTM(
                embedding_dim,
                hidden_dim,
                vocab_size,
                tagset_size,
                n_qubits=n_qubits,
                conv_kernel=conv_kernel,
                autoencoder_latent=autoencoder_latent,
                graph_arch=graph_arch,
            )
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        if isinstance(self.lstm, HybridQLSTM):
            tag_logits = self.lstm(sentence)
        else:
            lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
            tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["HybridQLSTM", "LSTMTagger"]
