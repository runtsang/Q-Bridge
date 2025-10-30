"""
HybridAutoencoder – quantum component
=====================================

Quantum routines that complement the classical encoder:

* `quantum_autoencoder_circuit` builds a swap‑test‑based variational autoencoder
  using `RealAmplitudes`.  A domain‑wall layer is optionally applied.
* `QuantumAutoencoder` returns a `SamplerQNN` that can be used as a quantum decoder.
* `QuantumEstimatorQNN` mirrors the classical `EstimatorNN` but uses a
  `StatevectorEstimator` to evaluate a single‑qubit observable.
* `HybridAutoencoder` stitches the two worlds together: a classical encoder feeds
  into the quantum decoder, and an optional quantum estimator is attached to the
  latent representation for regression on the latent space.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler, StatevectorEstimator
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN as QEstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import Parameter
import torch
from torch import nn


# --------------------------------------------------------------------------- #
# 1. Core quantum autoencoder circuit
# --------------------------------------------------------------------------- #
def quantum_autoencoder_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
    """
    Build a swap‑test‑based variational autoencoder.

    Parameters
    ----------
    num_latent : int
        Number of latent qubits that encode the classical latent vector.
    num_trash : int
        Number of auxiliary qubits used for the swap test.
    """
    total_qubits = num_latent + 2 * num_trash + 1
    qr = QuantumRegister(total_qubits, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)

    # Ansatz – a deep RealAmplitudes block
    ansatz = RealAmplitudes(num_latent + num_trash, reps=5)
    qc.compose(ansatz, range(0, num_latent + num_trash), inplace=True)

    # Swap‑test core
    auxiliary = num_latent + 2 * num_trash
    qc.h(auxiliary)
    for i in range(num_trash):
        qc.cswap(auxiliary, num_latent + i, num_latent + num_trash + i)
    qc.h(auxiliary)
    qc.measure(auxiliary, cr[0])

    return qc


def domain_wall(circuit: QuantumCircuit, start: int, end: int) -> QuantumCircuit:
    """
    Insert a domain‑wall (X) pattern on qubits [start, end).
    """
    for i in range(start, end):
        circuit.x(i)
    return circuit


# --------------------------------------------------------------------------- #
# 2. Quantum decoder (SamplerQNN)
# --------------------------------------------------------------------------- #
def QuantumAutoencoder(num_latent: int = 3, num_trash: int = 2) -> SamplerQNN:
    """
    Construct a quantum autoencoder as a SamplerQNN.

    Returns
    -------
    SamplerQNN
        Quantum neural network that maps latent parameters to reconstructed data.
    """
    algorithm_globals.random_seed = 42  # reproducibility
    sampler = StatevectorSampler()
    qc = quantum_autoencoder_circuit(num_latent, num_trash)

    # Optional domain wall for added expressivity
    dw_circ = domain_wall(QuantumCircuit(num_latent + 2 * num_trash), 0, num_latent + num_trash)
    qc.compose(dw_circ, range(num_latent + 2 * num_trash), inplace=True)

    def identity(x):
        return x

    return SamplerQNN(
        circuit=qc,
        input_params=[],
        weight_params=qc.parameters,
        interpret=identity,
        output_shape=2,
        sampler=sampler,
    )


# --------------------------------------------------------------------------- #
# 3. Quantum estimator (regression)
# --------------------------------------------------------------------------- #
def QuantumEstimatorQNN() -> QEstimatorQNN:
    """
    A single‑qubit variational estimator that returns a scalar prediction.
    """
    param_in = Parameter("x")
    param_w = Parameter("w")
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.ry(param_in, 0)
    qc.rx(param_w, 0)

    observable = SparsePauliOp.from_list([("Y", 1)])
    estimator = StatevectorEstimator()
    return QEstimatorQNN(
        circuit=qc,
        observables=observable,
        input_params=[param_in],
        weight_params=[param_w],
        estimator=estimator,
    )


# --------------------------------------------------------------------------- #
# 4. Hybrid wrapper
# --------------------------------------------------------------------------- #
class HybridAutoencoder(nn.Module):
    """
    A hybrid autoencoder that couples a classical encoder with a quantum decoder.

    The forward pass:
        1. Classical encoder maps `x` → latent vector (`z`).
        2. Latent vector is converted to a set of parameters for the quantum decoder.
        3. The quantum decoder (SamplerQNN) reconstructs the output.
    """
    def __init__(
        self,
        classical_encoder: nn.Module,
        quantum_decoder: SamplerQNN,
        latent_dim: int,
    ) -> None:
        super().__init__()
        self.encoder = classical_encoder
        self.decoder = quantum_decoder
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode to latent space
        z = self.encoder.encode(x)
        # Map latent to parameters for the quantum circuit
        # Here we simply treat each latent dimension as a separate weight
        # (requires quantum_decoder to have that many parameters)
        params = z.cpu().numpy().flatten()
        # Ensure the number of parameters matches the circuit
        if len(params)!= len(self.decoder.weight_params):
            raise ValueError("Mismatch between latent dim and quantum parameters.")
        # Evaluate quantum decoder
        q_outs = self.decoder(params)
        return torch.tensor(q_outs, dtype=x.dtype, device=x.device)


def HybridAutoencoderFactory(
    input_dim: int,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    bias: bool = True,
) -> HybridAutoencoder:
    """
    Convenience factory that builds a classical encoder and a quantum decoder
    and stitches them together.
    """
    from.ml_code import ClassicalAutoencoder  # local import to avoid circular dependency

    encoder = ClassicalAutoencoder(
        input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        bias=bias,
    )
    decoder = QuantumAutoencoder(num_latent=latent_dim)
    return HybridAutoencoder(encoder, decoder, latent_dim)


__all__ = [
    "quantum_autoencoder_circuit",
    "domain_wall",
    "QuantumAutoencoder",
    "QuantumEstimatorQNN",
    "HybridAutoencoder",
    "HybridAutoencoderFactory",
]
