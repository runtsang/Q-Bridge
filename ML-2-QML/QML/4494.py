"""
AutoencoderHybrid – quantum implementation.

This module implements a quantum‑classical hybrid autoencoder that
leverages Qiskit’s EstimatorQNN for a variational encoder, a
classical dense decoder, and an optional quantum LSTM (QLSTM) for
sequential latent processing.  The design mirrors the classical
module but replaces the encoder with a parameterised quantum circuit
and adds a hybrid expectation head akin to the QCNet hybrid layer.

Key components:
1. QuantumEncoder – a 1‑qubit variational circuit wrapped by
   EstimatorQNN.  The circuit is H → Ry(θ) → measure.
2. HybridFunction – a torch.autograd.Function that forwards the
   expectation value of a Pauli‑Z observable and implements a
   parameter‑shift gradient.
3. QLSTM – a minimal quantum LSTM cell that uses a 2‑qubit
   RealAmplitudes ansatz for each gate.  It is inspired by the
   QLSTM implementation in the seed but uses Qiskit primitives.
4. AutoencoderHybrid – combines the quantum encoder, optional
   quantum LSTM, and a classical decoder.

The module is fully importable and can be swapped with the classical
variant by simply changing the import path.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorEstimator, Sampler
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit import Aer

# ----------------------------------------------------------------------
# Quantum Encoder
# ----------------------------------------------------------------------
class QuantumEncoder(nn.Module):
    """
    1‑qubit variational encoder: H → Ry(θ) → measure.
    The θ parameter is trainable and is fed by the input tensor.
    """
    def __init__(self, backend=None, shots: int = 1024) -> None:
        super().__init__()
        self.backend = backend or Aer.get_backend("aer_simulator")
        self.shots = shots

        # Circuit definition
        self.theta = Parameter("theta")
        self.circuit = QuantumCircuit(1)
        self.circuit.h(0)
        self.circuit.ry(self.theta, 0)
        self.circuit.measure_all()

        # Estimator for expectation of Z
        self.estimator = StatevectorEstimator()
        self.estimator_qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=[SparsePauliOp.from_list([("Z", 1)])],
            input_params=[self.theta],
            weight_params=[],
            estimator=self.estimator,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: x is expected to be (batch, 1) and is interpreted
        as the value of θ.  The output is the expectation of Z.
        """
        # Ensure shape
        if x.ndim == 1:
            x = x.unsqueeze(-1)
        return self.estimator_qnn(x)


# ----------------------------------------------------------------------
# Hybrid Function – differentiable expectation
# ----------------------------------------------------------------------
class HybridFunction(torch.autograd.Function):
    """
    Wraps the quantum expectation value of a single‑qubit circuit
    and implements the parameter‑shift rule for gradients.
    """
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float = np.pi / 2) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        ctx.backend = Aer.get_backend("aer_simulator")
        # Run the circuit for each input value
        thetas = inputs.squeeze().tolist()
        results = []
        for theta in thetas:
            bound = {circuit.parameters[0]: theta}
            compiled = circuit.copy()
            compiled.bind_parameters(bound)
            state = Statevector.from_instruction(compiled)
            exp = state.expectation_value(SparsePauliOp.from_list([("Z", 1)]))
            results.append(exp)
        return torch.tensor(results, dtype=torch.float32, device=inputs.device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        shift = ctx.shift
        grad_inputs = torch.zeros_like(grad_output)
        for i, val in enumerate(grad_output):
            # Forward shift
            pos = HybridFunction.forward(
                torch.tensor([val.item() + shift], device=val.device),
                ctx.circuit,
                shift,
            )
            # Backward shift
            neg = HybridFunction.forward(
                torch.tensor([val.item() - shift], device=val.device),
                ctx.circuit,
                shift,
            )
            grad_inputs[i] = (pos - neg) * val
        return grad_inputs, None, None


# ----------------------------------------------------------------------
# Quantum LSTM (QLSTM) – minimal illustration
# ----------------------------------------------------------------------
class QLSTM(nn.Module):
    """
    Very small quantum LSTM cell that uses a 2‑qubit RealAmplitudes ansatz
    for each gate.  The cell is purely illustrative and not optimized
    for performance.
    """
    class QGate(nn.Module):
        def __init__(self, n_qubits: int = 2, reps: int = 1) -> None:
            super().__init__()
            self.n_qubits = n_qubits
            self.reps = reps
            self.ansatz = RealAmplitudes(n_qubits, reps=reps)
            # Parameters for the ansatz
            self.params = nn.Parameter(torch.randn(self.ansatz.num_parameters))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x contains concatenated hidden + cell states
            # We bind them to the ansatz parameters
            bound_params = {p: val for p, val in zip(self.ansatz.parameters, x.flatten(1))}
            qc = self.ansatz.copy()
            qc.bind_parameters(bound_params)
            # Measure all qubits
            meas = qc.measure_all()
            # Use statevector to compute expectation of Z on qubit 0
            state = Statevector.from_instruction(meas)
            exp = state.expectation_value(SparsePauliOp.from_list([("Z", 1)]))
            return torch.tensor(exp, dtype=torch.float32, device=x.device)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 2) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Four gates
        self.forget = self.QGate(n_qubits, reps=1)
        self.input = self.QGate(n_qubits, reps=1)
        self.update = self.QGate(n_qubits, reps=1)
        self.output = self.QGate(n_qubits, reps=1)

        # Linear layers to map classical vectors to gate parameters
        self.lin_f = nn.Linear(input_dim + hidden_dim, self.forget.ansatz.num_parameters)
        self.lin_i = nn.Linear(input_dim + hidden_dim, self.input.ansatz.num_parameters)
        self.lin_u = nn.Linear(input_dim + hidden_dim, self.update.ansatz.num_parameters)
        self.lin_o = nn.Linear(input_dim + hidden_dim, self.output.ansatz.num_parameters)

    def forward(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(self.lin_f(combined)))
            i = torch.sigmoid(self.input(self.lin_i(combined)))
            u = torch.tanh(self.update(self.lin_u(combined)))
            o = torch.sigmoid(self.output(self.lin_o(combined)))
            cx = f * cx + i * u
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        return torch.cat(outputs, dim=0), (hx, cx)

    def _init_states(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )


# ----------------------------------------------------------------------
# Hybrid Autoencoder
# ----------------------------------------------------------------------
class AutoencoderHybrid(nn.Module):
    """
    Quantum‑classical autoencoder that combines:
    - QuantumEncoder (EstimatorQNN) for latent representation.
    - Optional QLSTM for sequential latent processing.
    - Classical dense decoder for reconstruction.
    """
    def __init__(
        self,
        input_dim: int = 2,
        latent_dim: int = 1,
        n_qubits: int = 1,
        use_lstm: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = QuantumEncoder(backend=Aer.get_backend("aer_simulator"), shots=1024)
        self.use_lstm = use_lstm
        if use_lstm:
            self.lstm = QLSTM(input_dim=latent_dim, hidden_dim=latent_dim, n_qubits=n_qubits)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantum encoding
        z = self.encoder(x)
        if self.use_lstm:
            z, _ = self.lstm(z.unsqueeze(1))
            z = z.squeeze(1)
        # Classical decoding
        return self.decoder(z)

    def reconstruction_loss(self, x: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
        # Mean squared error between raw input and reconstruction
        return F.mse_loss(recon, x, reduction="mean")


__all__ = ["AutoencoderHybrid", "QuantumEncoder", "HybridFunction", "QLSTM"]
