"""Quantum‑enhanced LSTM with a quantum autoencoder and a quantum expectation head.

This module defines :class:`GenQLSTM` that mirrors the classical counterpart
but replaces the linear gates of the LSTM with small variational
circuits.  The sequence is first projected into a latent space by a
parameterised quantum autoencoder (implemented with Qiskit), then
processed by a quantum LSTM.  The final classification head is a
parameterised quantum circuit that returns the expectation value of
Pauli‑Z, mimicking the behaviour of a quantum measurement.

The implementation is fully differentiable thanks to the custom
``HybridFunction`` that implements the parameter‑shift rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchquantum as tq
import torchquantum.functional as tqf
import qiskit
from qiskit import QuantumCircuit as QiskitCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import Sampler as StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN as QiskitSamplerQNN


def quantum_autoencoder(num_latent: int = 3, num_trash: int = 2) -> QiskitCircuit:
    """Build a simple quantum autoencoder circuit for a given latent size."""
    qr = qiskit.QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = qiskit.ClassicalRegister(1, "c")
    circuit = QiskitCircuit(qr, cr)
    # Ansatz
    ansatz = qiskit.circuit.library.RealAmplitudes(num_latent + num_trash, reps=5)
    circuit.append(ansatz, range(0, num_latent + num_trash))
    circuit.barrier()
    # Swap test
    aux = num_latent + 2 * num_trash
    circuit.h(aux)
    for i in range(num_trash):
        circuit.cswap(aux, num_latent + i, num_latent + num_trash + i)
    circuit.h(aux)
    circuit.measure(aux, cr[0])
    return circuit


class QuantumCircuitWrapper:
    """Wrapper around a parametrised two‑qubit circuit executed on Aer."""
    def __init__(self, n_qubits: int, backend, shots: int) -> None:
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        all_qubits = list(range(n_qubits))
        self.theta = qiskit.circuit.Parameter("theta")
        self._circuit.h(all_qubits)
        self._circuit.barrier()
        self._circuit.ry(self.theta, all_qubits)
        self._circuit.measure_all()
        self.backend = backend
        self.shots = shots

    def run(self, thetas: torch.Tensor) -> torch.Tensor:
        """Execute the parametrised circuit for the provided angles."""
        compiled = qiskit.transpile(self._circuit, self.backend)
        qobj = qiskit.assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: float(theta)} for theta in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        # expectation value of Z
        counts = sum(result.values())
        probs = {k: v / counts for k, v in result.items()}
        exp = sum(int(k, 2) * p for k, p in probs.items())
        return torch.tensor([exp], dtype=torch.float32)


class HybridFunction(torch.autograd.Function):
    """Differentiable interface between PyTorch and the quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuitWrapper, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        expectation_z = ctx.circuit.run(inputs.detach().cpu().numpy())
        result = torch.tensor(expectation_z, device=inputs.device)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        grad = []
        for val in inputs.detach().cpu().numpy():
            right = ctx.circuit.run(np.array([val + shift]))
            left = ctx.circuit.run(np.array([val - shift]))
            grad.append(right - left)
        grad = torch.tensor(grad, device=inputs.device, dtype=torch.float32)
        return grad * grad_output, None, None


class Hybrid(nn.Module):
    """Hybrid layer that forwards activations through a quantum circuit."""
    def __init__(self, n_qubits: int, backend, shots: int, shift: float) -> None:
        super().__init__()
        self.quantum_circuit = QuantumCircuitWrapper(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs, self.quantum_circuit, self.shift)


class QuantumLayer(tq.QuantumModule):
    def __init__(self, n_wires: int) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
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
                tqf.cnot(qdev, wires=[wire, 0])
            else:
                tqf.cnot(qdev, wires=[wire, wire + 1])
        return self.measure(qdev)


class QuantumLSTM(nn.Module):
    """LSTM cell where gates are realised by small quantum circuits."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.forget = QuantumLayer(n_qubits)
        self.input = QuantumLayer(n_qubits)
        self.update = QuantumLayer(n_qubits)
        self.output = QuantumLayer(n_qubits)
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
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


@dataclass
class GenQLSTMConfig:
    """Configuration for the hybrid GenQLSTM model."""
    embedding_dim: int
    hidden_dim: int
    vocab_size: int
    tagset_size: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    shift: float = 0.0
    n_qubits: int = 4


class GenQLSTM(nn.Module):
    """Quantum LSTM with quantum autoencoder preprocessing and quantum expectation head."""
    def __init__(self, config: GenQLSTMConfig) -> None:
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_dim)
        # Quantum autoencoder circuit (simple demo)
        self.autoencoder_circuit = quantum_autoencoder(num_latent=config.latent_dim)
        self.autoencoder_sampler = StatevectorSampler()
        # Simple linear decoder for demo; a full VAE would be used in practice
        self.decoder = nn.Linear(config.latent_dim, config.embedding_dim)
        self.lstm = QuantumLSTM(
            config.latent_dim,
            config.hidden_dim,
            n_qubits=config.n_qubits,
        )
        self.classifier = nn.Linear(config.hidden_dim, config.tagset_size)
        backend = qiskit.Aer.get_backend("aer_simulator")
        self.hybrid = Hybrid(config.tagset_size, backend, shots=100, shift=config.shift)

    def _quantum_encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode classical data into a latent vector using the quantum autoencoder.
        For brevity, we use the sampler to get expectation values of a simple circuit.
        """
        # Flatten input to 1‑D array for the sampler
        flat = x.view(-1, x.size(-1))
        # Run the sampler for each data point
        latent = []
        for sample in flat:
            # Sample expects a 1‑D numpy array of parameters
            res = self.autoencoder_sampler.run(sample.cpu().numpy())
            latent.append(res)
        latent = torch.stack(latent, dim=0).to(x.device).float()
        return latent

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)  # (seq_len, batch, embed)
        seq_len, batch, embed = embeds.shape
        flat = embeds.reshape(seq_len * batch, embed)
        # Quantum encoding
        latent = self._quantum_encode(flat)  # (seq_len*batch, latent_dim)
        latent = latent.reshape(seq_len, batch, -1)
        lstm_out, _ = self.lstm(latent)
        logits = self.classifier(lstm_out)
        probs = self.hybrid(logits)
        return F.log_softmax(probs, dim=-1)


__all__ = ["GenQLSTM", "GenQLSTMConfig"]
