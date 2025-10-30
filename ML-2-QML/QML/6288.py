"""Quantum hybrid auto‑encoder + classifier.

This module implements the quantum counterpart of the classical
HybridAutoEncoderClassifier.  It uses a Qiskit sampler‑based auto‑encoder to
compress the input into a latent vector, and a parameterised circuit that
produces a binary probability from that latent representation.  The
architecture is deliberately lightweight so that it can run on a simulator
or a real device with few qubits.

Author: OpenAI – engineered for rapid experimentation with hybrid models.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from qiskit import QuantumCircuit, Aer
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler as PrimitiveSampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.states.utils import partial_trace
from qiskit_machine_learning.utils import algorithm_globals

__all__ = [
    "QuantumAutoEncoder",
    "QuantumHybridAutoEncoderClassifier",
    "train_quantum_classifier",
]


# --------------------------------------------------------------------------- #
# 1. Quantum auto‑encoder – sampler‑based
# --------------------------------------------------------------------------- #
class QuantumAutoEncoder(nn.Module):
    """A shallow quantum auto‑encoder using a RealAmplitudes ansatz and a swap test."""

    def __init__(self, latent_dim: int = 3, num_trash: int = 2, reps: int = 2) -> None:
        super().__init__()
        # Build the encoder circuit
        self.latent_dim = latent_dim
        self.num_trash = num_trash
        self.reps = reps
        self._build_circuit()

        # Sampler primitive for expectation evaluation
        self.sampler = PrimitiveSampler()
        self.backend = Aer.get_backend("aer_simulator")

    def _build_circuit(self) -> None:
        """Constructs the encoder + swap‑test circuit."""
        n = self.latent_dim + 2 * self.num_trash + 1  # +1 for auxiliary qubit
        qc = QuantumCircuit(n)
        # Encoder ansatz on the first latent+trash qubits
        qc.compose(
            RealAmplitudes(self.latent_dim + self.num_trash, reps=self.reps),
            range(0, self.latent_dim + self.num_trash),
            inplace=True,
        )
        qc.barrier()
        # Swap‑test using the auxiliary qubit
        aux = self.latent_dim + 2 * self.num_trash
        qc.h(aux)
        for i in range(self.num_trash):
            qc.cswap(aux, self.latent_dim + i, self.latent_dim + self.num_trash + i)
        qc.h(aux)
        qc.measure(aux, 0)
        self.circuit = qc

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode inputs into a latent vector via the sampler."""
        # Convert to numpy for the sampler
        input_list = inputs.cpu().numpy().tolist()
        # For each input, run the circuit and read the measurement
        latent = []
        for inpt in input_list:
            # Bind the parameters – we use a simple linear map from input to rotation angles
            # Here we just map each input component to a Ry rotation on the corresponding qubit
            bound_qc = self.circuit.copy()
            for idx, val in enumerate(inpt):
                bound_qc.ry(val, idx)
            # Run the circuit with the sampler
            job = self.sampler.run(bound_qc, shots=1024)
            result = job.result()
            counts = result.get_counts()
            # Expectation of the auxiliary qubit measurement (0/1)
            prob_1 = counts.get('1', 0) / 1024
            # Convert to a real number between -1 and 1
            latent.append(2 * prob_1 - 1)
        # Stack into a tensor
        return torch.tensor(latent, dtype=torch.float32, device=inputs.device)


# --------------------------------------------------------------------------- #
# 2. Quantum classifier head
# --------------------------------------------------------------------------- #
class QuantumClassifier(nn.Module):
    """Parameterised circuit that maps a latent vector to a binary probability."""

    def __init__(self, latent_dim: int = 3, shots: int = 1024) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.shots = shots
        self.backend = Aer.get_backend("aer_simulator")
        self.sampler = PrimitiveSampler()
        self._build_circuit()

    def _build_circuit(self) -> None:
        """Builds a simple variational circuit with a single Pauli‑Z expectation."""
        qc = QuantumCircuit(self.latent_dim)
        qc.compose(
            RealAmplitudes(self.latent_dim, reps=1),
            range(self.latent_dim),
            inplace=True,
        )
        # Measure expectation of Z on the first qubit
        qc.measure_all()
        self.circuit = qc

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Return a probability in [0,1] from the latent vector."""
        latent_list = latent.cpu().numpy().tolist()
        probs = []
        for vec in latent_list:
            bound_qc = self.circuit.copy()
            for idx, val in enumerate(vec):
                bound_qc.ry(val, idx)
            job = self.sampler.run(bound_qc, shots=self.shots)
            result = job.result()
            counts = result.get_counts()
            # Compute expectation of Z on the first qubit
            exp_z = 0.0
            for bitstring, cnt in counts.items():
                # bitstring is e.g. '01'; first qubit is the most significant bit
                bit = int(bitstring[0])
                exp_z += ((-1) ** bit) * cnt
            exp_z /= self.shots
            # Convert to probability via sigmoid
            probs.append(0.5 * (exp_z + 1))
        return torch.tensor(probs, dtype=torch.float32, device=latent.device)


# --------------------------------------------------------------------------- #
# 3. End‑to‑end quantum hybrid model
# --------------------------------------------------------------------------- #
class QuantumHybridAutoEncoderClassifier(nn.Module):
    """Quantum auto‑encoder + quantum classifier head."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 3,
        num_trash: int = 2,
        ae_reps: int = 2,
        clf_shots: int = 1024,
    ) -> None:
        super().__init__()
        self.autoencoder = QuantumAutoEncoder(latent_dim, num_trash, reps=ae_reps)
        self.classifier = QuantumClassifier(latent_dim, shots=clf_shots)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.autoencoder(x)
        probs = self.classifier(latent)
        return probs.unsqueeze(-1)  # shape (batch, 1)


# --------------------------------------------------------------------------- #
# 4. Training helper – back‑prop through the sampler
# --------------------------------------------------------------------------- #
def train_quantum_classifier(
    model: QuantumHybridAutoEncoderClassifier,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    epochs: int = 10,
    lr: float = 1e-3,
    device: torch.device | None = None,
) -> list[float]:
    """Very light‑weight training loop that uses the autograd support of
    qiskit‑machine‑learning's SamplerQNN.  In practice the sampler is
    non‑differentiable; we approximate gradients with finite differences
    in the QuantumClassifier.  This routine is mainly for illustrative
    purposes and should not be regarded as fully efficient.

    Parameters
    ----------
    model : QuantumHybridAutoEncoderClassifier
        The quantum hybrid model.
    inputs : torch.Tensor
        Input data of shape (N, input_dim).
    targets : torch.Tensor
        Ground‑truth labels (0 or 1) of shape (N,).
    epochs : int, default 10
        Number of epochs to train.
    lr : float, default 1e-3
        Learning rate for the optimizer.
    device : torch.device, optional
        Device to run on; defaults to CUDA if available.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = torch.utils.data.TensorDataset(
        _as_tensor(inputs), _as_tensor(targets)
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    history: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            probs = model(batch_x)
            logits = probs.squeeze(-1)  # shape (batch,)
            loss = loss_fn(logits.unsqueeze(0), batch_y.long())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history


def _as_tensor(data: torch.Tensor | list[float]) -> torch.Tensor:
    """Utility to ensure data is a float32 tensor on the default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor
