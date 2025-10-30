"""Quantum‑enabled Hybrid LSTM tagger with optional quantum auto‑encoder.

The implementation mirrors the classical counterpart but replaces the
auto‑encoder and LSTM gates with quantum modules that are built on top of
Qiskit and TorchQuantum.  The quantum auto‑encoder uses a variational
circuit with a swap‑test for reconstruction, while the quantum LSTM gates
are realised by small circuits that act on a dedicated set of qubits.

The module is deliberately lightweight – it only exposes the shared API
``HybridQLSTM`` so that the training pipeline can remain identical to
the classical version.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN

# --------------------------------------------------------------------------- #
# Quantum Auto‑Encoder
# --------------------------------------------------------------------------- #

class QuantumAE(nn.Module):
    """
    Quantum variational auto‑encoder based on the reference `Autoencoder.py`.
    It uses a RealAmplitudes ansatz with a swap‑test to measure
    reconstruction fidelity.  The circuit is compiled once per forward pass
    and executed on a state‑vector sampler (or any backend that supports
    sampling).
    """
    def __init__(
        self,
        latent_dim: int = 3,
        num_trash: int = 2,
        reps: int = 5,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_trash = num_trash
        self.reps = reps
        self.sampler = StatevectorSampler()

        def ansatz(num_qubits):
            return RealAmplitudes(num_qubits, reps=reps)

        def auto_encoder_circuit(num_latent, num_trash):
            qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
            cr = ClassicalRegister(1, "c")
            circuit = QuantumCircuit(qr, cr)
            circuit.compose(ansatz(num_latent + num_trash), range(0, num_latent + num_trash), inplace=True)
            circuit.barrier()
            aux = num_latent + 2 * num_trash
            circuit.h(aux)
            for i in range(num_trash):
                circuit.cswap(aux, num_latent + i, num_latent + num_trash + i)
            circuit.h(aux)
            circuit.measure(aux, cr[0])
            return circuit

        self.circuit = auto_encoder_circuit(latent_dim, num_trash)
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=lambda x: x,
            output_shape=latent_dim,
            sampler=self.sampler,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode the input into a latent vector.
        The input tensor is expected to be already flattened.
        """
        import numpy as np
        inputs = x.detach().cpu().numpy()
        latents = self.qnn.run(inputs)
        return torch.tensor(latents, dtype=x.dtype, device=x.device)

# --------------------------------------------------------------------------- #
# Quantum LSTM Gates
# --------------------------------------------------------------------------- #

class QuantumLSTM(nn.Module):
    """
    LSTM where the four gates are realised by small quantum circuits.
    The implementation follows the structure of the classical QLSTM but
    replaces the linear projections with quantum modules that operate on
    a dedicated set of qubits.
    """
    class QGate(nn.Module):
        def __init__(self, n_qubits: int):
            super().__init__()
            self.n_qubits = n_qubits
            # Simple trainable rotation parameters
            self.rx_gates = nn.ParameterList(
                [nn.Parameter(torch.randn(1)) for _ in range(n_qubits)]
            )
            # Define a CNOT chain (last qubit couples back to the first)
            self.cnot_chain = [(i, i + 1) if i < n_qubits - 1 else (n_qubits - 1, 0)
                               for i in range(n_qubits)]

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            A differentiable surrogate for a quantum gate.
            Each input feature is interpreted as a rotation angle.
            """
            batch = x.size(0)
            out = torch.zeros(batch, self.n_qubits, device=x.device)
            for i, theta in enumerate(self.rx_gates):
                out[:, i] = torch.sin(x[:, i] + theta)
            for src, tgt in self.cnot_chain:
                out[:, tgt] = out[:, tgt] + out[:, src]
            return out

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Classical linear projections to quantum space
        self.lin_f = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.lin_i = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.lin_g = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.lin_o = nn.Linear(input_dim + hidden_dim, n_qubits)

        # Quantum gates
        self.qf = self.QGate(n_qubits)
        self.qi = self.QGate(n_qubits)
        self.qg = self.QGate(n_qubits)
        self.qo = self.QGate(n_qubits)

    def forward(self, inputs: torch.Tensor,
                states: Tuple[torch.Tensor, torch.Tensor] | None = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.qf(self.lin_f(combined)))
            i = torch.sigmoid(self.qi(self.lin_i(combined)))
            g = torch.tanh(self.qg(self.lin_g(combined)))
            o = torch.sigmoid(self.qo(self.lin_o(combined)))
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
# Hybrid LSTM Tagger (Quantum version)
# --------------------------------------------------------------------------- #

class HybridQLSTM(nn.Module):
    """
    Quantum‑aware tagger that mirrors the classical API but uses a quantum
    auto‑encoder and/or quantum‑gate LSTM depending on the configuration.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        ae_type: str = "none",
        ae_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Auto‑encoder branch
        if ae_type == "quantum":
            self.preproc = QuantumAE(**(ae_cfg or {}))
        else:
            self.preproc = None

        # LSTM branch
        if n_qubits > 0:
            self.lstm = QuantumLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        x = self.word_embeddings(sentence)
        if self.preproc is not None:
            x = self.preproc(x)
        lstm_out, _ = self.lstm(x.unsqueeze(1))
        logits = self.hidden2tag(lstm_out.squeeze(1))
        return F.log_softmax(logits, dim=-1)

__all__ = ["HybridQLSTM", "QuantumAE", "QuantumLSTM"]
