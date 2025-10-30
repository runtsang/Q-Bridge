"""Quantum‑enabled hybrid kernel, auto‑encoder, and LSTM model.

The quantum variant of :class:`HybridKernelModel` uses TorchQuantum for
variational kernels and LSTM gates, and Qiskit + Qiskit‑Machine‑Learning for a
variational auto‑encoder and a parameter‑shift classifier.  The public API
mirrors the classical implementation, enabling transparent switching between
back‑ends.

"""

from __future__ import annotations

from typing import Iterable, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torchquantum.functional import MeasureAll, cnot

from qiskit import QuantumCircuit, Aer
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.utils import algorithm_globals
from qiskit.quantum_info import SparsePauliOp

# --------------------------------------------------------------------------- #
# 1. Quantum kernel
# --------------------------------------------------------------------------- #
class QuantumKernel(tq.QuantumModule):
    """Variational RBF‑style kernel implemented with TorchQuantum."""
    def __init__(self, n_wires: int = 4, depth: int = 1) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.depth = depth
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(self.n_wires)]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.q_device.reset_states(x.shape[0])
        self.ansatz(self.q_device, x)
        for i in reversed(range(self.n_wires)):
            cnot(self.q_device, wires=[i, (i + 1) % self.n_wires])
        self.ansatz(self.q_device, -y)
        for i in reversed(range(self.n_wires)):
            cnot(self.q_device, wires=[i, (i + 1) % self.n_wires])
        return torch.abs(self.q_device.states.view(-1)[0])

def quantum_kernel_matrix(a: Iterable[torch.Tensor], b: Iterable[torch.Tensor]) -> np.ndarray:
    kernel = QuantumKernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# --------------------------------------------------------------------------- #
# 2. Quantum auto‑encoder
# --------------------------------------------------------------------------- #
class QuantumAutoencoder(tq.QuantumModule):
    """Variational auto‑encoder built with a RealAmplitudes ansatz and a swap‑test."""
    def __init__(self, num_latent: int, num_trash: int = 2) -> None:
        super().__init__()
        algorithm_globals.random_seed = 42
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.circuit = self._build_circuit()
        self.sampler = Aer.get_backend("aer_simulator")
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=lambda x: x,
            output_shape=2,
            sampler=self.sampler,
        )

    def _build_circuit(self) -> QuantumCircuit:
        qr = QuantumCircuit(self.num_latent + 2 * self.num_trash + 1, 1)
        ansatz = RealAmplitudes(self.num_latent + self.num_trash, reps=5)
        qr.compose(ansatz, range(self.num_latent + self.num_trash), inplace=True)
        qr.barrier()
        aux = self.num_latent + 2 * self.num_trash
        qr.h(aux)
        for i in range(self.num_trash):
            qr.cswap(aux, self.num_latent + i, self.num_latent + self.num_trash + i)
        qr.h(aux)
        qr.measure(aux, 0)
        return qr

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the sampler output for the latent representation."""
        return self.qnn.forward(x)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """In this toy example the decoder is identical to the encoder."""
        return self.encode(latents)

# --------------------------------------------------------------------------- #
# 3. Quantum classifier
# --------------------------------------------------------------------------- #
class QuantumClassifier(tq.QuantumModule):
    """Parameter‑shift classifier built with a Qiskit circuit."""
    def __init__(self, num_qubits: int, depth: int) -> None:
        super().__init__()
        self.num_qubits = num_qubits
        self.depth = depth
        self.circuit, self.encoding, self.weights, self.observables = self._build_circuit()
        self.sampler = Aer.get_backend("aer_simulator")
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=self.encoding,
            weight_params=self.weights,
            interpret=lambda x: x,
            output_shape=len(self.observables),
            sampler=self.sampler,
        )

    def _build_circuit(self) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
        encoding = ParameterVector("x", self.num_qubits)
        weights = ParameterVector("theta", self.num_qubits * self.depth)
        circuit = QuantumCircuit(self.num_qubits)
        for param, qubit in zip(encoding, range(self.num_qubits)):
            circuit.rx(param, qubit)
        idx = 0
        for _ in range(self.depth):
            for qubit in range(self.num_qubits):
                circuit.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(self.num_qubits - 1):
                circuit.cz(qubit, qubit + 1)
        observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (self.num_qubits - i - 1))
            for i in range(self.num_qubits)
        ]
        return circuit, list(encoding), list(weights), observables

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.qnn.forward(x)

# --------------------------------------------------------------------------- #
# 4. Quantum LSTM (from reference pair 3)
# --------------------------------------------------------------------------- #
class QLSTM(tq.QuantumModule):
    """LSTM cell where gates are realised by small quantum circuits."""
    class QLayer(tq.QuantumModule):
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
            self.measure = MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            for wire in range(self.n_wires):
                if wire == self.n_wires - 1:
                    cnot(qdev, wires=[wire, 0])
                else:
                    cnot(qdev, wires=[wire, wire + 1])
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

    def forward(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None = None):
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

# --------------------------------------------------------------------------- #
# 5. Hybrid model
# --------------------------------------------------------------------------- #
class HybridKernelModel(tq.QuantumModule):
    """Unified quantum/classical hybrid model.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input feature vectors.
    latent_dim : int, default 32
        Size of the latent space in the quantum auto‑encoder.
    hidden_dims : Tuple[int, int], default (128, 64)
        Hidden layer sizes for the quantum auto‑encoder (used only in the
        classical branch).
    dropout : float, default 0.1
        Drop‑out probability in the quantum auto‑encoder (used only in the
        classical branch).
    kernel_n_wires : int, default 4
        Number of wires for the variational kernel.
    depth : int, default 2
        Depth of the feed‑forward classifier.
    n_qubits : int, default 4
        Number of qubits used if a quantum LSTM is requested.
    use_quantum_lstm : bool, default False
        Whether to replace the classical LSTM with a quantum variant.
    """
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: Tuple[int, int] = (128, 64),
        dropout: float = 0.1,
        kernel_n_wires: int = 4,
        depth: int = 2,
        n_qubits: int = 4,
        use_quantum_lstm: bool = False,
    ) -> None:
        super().__init__()
        self.kernel = QuantumKernel(n_wires=kernel_n_wires)
        self.autoencoder = QuantumAutoencoder(num_latent=latent_dim)
        self.classifier = QuantumClassifier(num_qubits=input_dim, depth=depth)
        if use_quantum_lstm:
            self.lstm = QLSTM(input_dim, hidden_dim=latent_dim, n_qubits=n_qubits)
        else:
            self.lstm = None

    def kernel_matrix(self, X: Iterable[torch.Tensor], Y: Iterable[torch.Tensor]) -> np.ndarray:
        """Return the Gram matrix between two collections of tensors."""
        return np.array([[self.kernel(x, y).item() for y in Y] for x in X])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode, classify and optionally run through an LSTM."""
        z = self.autoencoder.encode(x)
        logits = self.classifier(z)
        if self.lstm is not None:
            lstm_out, _ = self.lstm(z.unsqueeze(0))
            logits = self.classifier(lstm_out.squeeze(0))
        return logits

__all__ = [
    "HybridKernelModel",
    "QuantumKernel",
    "QuantumAutoencoder",
    "QuantumClassifier",
    "QLSTM",
]
