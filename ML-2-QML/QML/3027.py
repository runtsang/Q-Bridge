"""
Quantum implementation of the hybrid LSTM.

The quantum module provides a small variational circuit that
implements each LSTM gate.  It also exposes a SamplerQNN that
produces a probability distribution over two basis states,
mirroring the classical approximation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN as QiskitSamplerQNN
from qiskit.primitives import StatevectorSampler as StateSampler

# --------------------------------------------------------------------------- #
#  Variational quantum gate
# --------------------------------------------------------------------------- #
class QGate(tq.QuantumModule):
    """
    Gate that maps a real vector of length ``n_qubits`` into a
    probability vector over the computational basis of the same size.
    The underlying circuit consists of parameterised RX rotations,
    followed by a chain of CNOTs that entangle the qubits.
    """
    def __init__(self, n_qubits: int) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        # Encoder that injects the classical vector into the circuit
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "rx", "wires": [i]}
                for i in range(n_qubits)
            ]
        )
        # Trainable rotation gates
        self.params = nn.ModuleList(
            [tq.RX(has_params=True, trainable=True) for _ in range(n_qubits)]
        )
        # Measurement
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (..., n_qubits).  Each element is a rotation angle.
        Returns
        -------
        torch.Tensor
            Probabilities over the basis states.
        """
        dev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=x.shape[0], device=x.device)
        self.encoder(dev, x)
        for gate, wire in zip(self.params, range(self.n_qubits)):
            gate(dev, wires=wire)
        # Entangle via a linear chain of CNOTs
        for i in range(self.n_qubits - 1):
            tqf.cnot(dev, wires=[i, i + 1])
        return self.measure(dev)


# --------------------------------------------------------------------------- #
#  Sampler QNN
# --------------------------------------------------------------------------- #
class SamplerQNN(nn.Module):
    """
    Quantum sampler that implements the circuit from the reference
    pair.  It is compatible with Qiskit Machine Learning's
    :class:`~qiskit_machine_learning.neural_networks.SamplerQNN`.
    """
    def __init__(self) -> None:
        super().__init__()
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
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
        return qc

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that samples from the quantum circuit.

        Parameters
        ----------
        inputs : torch.Tensor
            Shape (..., 2) – the two input angles.
        Returns
        -------
        torch.Tensor
            Probabilities over the two output basis states.
        """
        # Use the StatevectorSampler to evaluate the circuit
        sampler = StateSampler()
        # Convert torch tensor to numpy for the sampler
        angles = inputs.detach().cpu().numpy()
        # Build parameter dictionary
        param_dict = {
            "input[0]": angles[..., 0],
            "input[1]": angles[..., 1],
        }
        # Sample probabilities
        probs = sampler.run(
            self.circuit,
            parameter_binds=[param_dict],
            shots=0,  # use statevector
        )
        return torch.tensor(probs, device=inputs.device, dtype=inputs.dtype)


# --------------------------------------------------------------------------- #
#  Hybrid LSTM (Quantum)
# --------------------------------------------------------------------------- #
class HybridQLSTMQuantum(nn.Module):
    """
    Quantum‑enhanced LSTM that forwards each gate through
    :class:`QGate`.  The linear layers that feed the gates are
    identical to the classical version but produce qubit‑sized
    vectors that are interpreted by the quantum circuit.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        self.forget = QGate(n_qubits)
        self.input = QGate(n_qubits)
        self.update = QGate(n_qubits)
        self.output = QGate(n_qubits)

        self.forget_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.input_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.update_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.output_lin = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = states
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(self.forget_lin(combined)))
            i = torch.sigmoid(self.input(self.input_lin(combined)))
            g = torch.tanh(self.update(self.update_lin(combined)))
            o = torch.sigmoid(self.output(self.output_lin(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        return torch.cat(outputs, dim=0), (hx, cx)


# --------------------------------------------------------------------------- #
#  LSTM Tagger (Quantum)
# --------------------------------------------------------------------------- #
class LSTMTaggerQuantum(nn.Module):
    """
    Sequence tagging model that uses the quantum hybrid LSTM.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int,
    ) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = HybridQLSTMQuantum(embedding_dim, hidden_dim, n_qubits=n_qubits)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return torch.log_softmax(logits, dim=1)


__all__ = ["QGate", "SamplerQNN", "HybridQLSTMQuantum", "LSTMTaggerQuantum"]
